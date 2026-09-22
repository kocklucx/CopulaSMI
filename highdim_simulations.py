import torch
import math
from scipy import stats
from copula_smi import SMI_VB
import pickle
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import matplotlib.pyplot as plt
import string
from collections import OrderedDict
from scipy.stats import gaussian_kde

def normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def psi_to_omega(psi):
    psi = psi.flatten()
    params = torch.cat([torch.exp(psi[0:1]), psi[1:]])
    params = torch.tanh(params)
    return torch.outer(params, params) + torch.diag(1 - params**2)

def log_pdf_copula(u,psi):
    # Gaussian copula
    u = u.clamp(1e-5,1-1e-5)
    omega = psi_to_omega(psi)
    u = torch.distributions.normal.Normal(0, 1).icdf(u)
    return (-.5*(u[:,None,:]@torch.matmul(torch.inverse(omega)-torch.eye(d),u[:,:,None])).squeeze()-0.5*torch.log(torch.det(omega))).sum()

def log_pdf_marginals(y,eta):
    mu = eta[[(2*j) for j in range(d)]].flatten()
    sigma = eta[[2*j+1 for j in range(d)]].exp().flatten()
    l = torch.distributions.normal.Normal(mu,sigma).log_prob(y)
    return l.sum()

def cdfs_marginals(y,eta):
    mu = eta[[(2*j) for j in range(d)]].flatten()
    sigma = eta[[2*j+1 for j in range(d)]].exp().flatten()
    cdf = torch.distributions.normal.Normal(mu,sigma).cdf(y)
    return cdf

def log_prior(psi,eta):
    # psi
    psi = psi.flatten()
    p_psi = (torch.distributions.normal.Normal(loc=0,scale=2).log_prob(psi[1:])).sum()
    p_psi += torch.distributions.half_normal.HalfNormal(2).log_prob(torch.exp(psi[0]))+psi[0]
    # eta
    mu = eta[[(2*j) for j in range(d)]].flatten()
    sigma = eta[[2*j+1 for j in range(d)]].exp().flatten()
    p_mu = torch.distributions.normal.Normal(loc=0,scale=2.5).log_prob(mu).sum()
    p_sigma = (torch.distributions.half_normal.HalfNormal(10).log_prob(sigma)+sigma.log()).sum() 
    return p_psi + p_mu + p_sigma

def score(vb):
    # utility function
    num_mc_samples = 5000
    result = []
    for _ in range(num_mc_samples):
        z, psi, eta, eta_tilde = vb.single_sample()   
        sigma = eta[[2*j+1 for j in range(d)]].exp().flatten()
        omega =  psi_to_omega(psi)
        cov_matrix = torch.diag(sigma)@omega@torch.diag(sigma)
        weights = torch.linalg.solve(cov_matrix,torch.ones(d))
        weights /= weights.sum()
        result.append(-(weights*y).sum(axis=1).var().item())
    return torch.tensor(result).mean()
    
def sample_from_copula(omega,n):
    return normal_cdf(torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(d),covariance_matrix=omega).sample((n,)))

block_size = 5
d = 3*block_size
n = 1000
iterations_training = 50000

dim_psi, dim_eta = d,int(2*d)

true_loadings = torch.tensor([0.7,0.7,0.7,0.75,0.75,0.75,0.8,.8,.8,0.85,0.85,0.85,0.9,0.9,0.9])

psi_true = torch.atanh(true_loadings)
psi_true[0] = torch.log(psi_true[0])
omega_true = psi_to_omega(psi_true)
    
marginals = []
for j in range(block_size):
    marginals += [stats.norm(0,1),stats.skewnorm(loc=0,scale=1,a=-5),stats.Mixture([stats.Normal(mu=-1, sigma=0.4), stats.Normal(mu=0., sigma=0.2)], weights=[0.6, 0.4])]

if False:
    u_true = sample_from_copula(omega_true,n)
    y = torch.empty(size=u_true.shape)
    for j in range(d):
        try:
            y[:,j] = torch.tensor(marginals[j].ppf(u_true[:,j]))
        except:
            y[:,j] = torch.tensor(marginals[j].icdf(u_true[:,j]))
            
    pickle.dump(y,open('data_highdimensional.p','wb'))
else:
    y =  pickle.load(open('data_highdimensional.p', 'rb'))

vb_conventional = SMI_VB(torch.ones(d),dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
vb_conventional.train(iterations_training)
pickle.dump(vb_conventional,open('results/vb_conventional.p','wb'))
print('conventional done')

vb_cut = SMI_VB(torch.zeros(d),dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
vb_cut.train(iterations_training)
pickle.dump(vb_cut,open('results/vb_cut.p','wb'))
print('cut done')

bounds = torch.stack([torch.zeros(int(d)), torch.ones(int(d))]).to(torch.double)

num_model = 0
candidates = []
scores = []
for _ in range(10):
    gamma = torch.rand(d)
    vb = SMI_VB(gamma,dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
    vb.train(iterations_training)
    pickle.dump(vb,open('results/smi_'+str(num_model)+'.p','wb'))
    scores.append(score(vb))
    candidates.append(gamma)
    print(str(num_model)+' done')
    num_model += 1
    
candidates = torch.vstack(candidates).to(torch.double)
scores = torch.vstack(scores).to(torch.double)
pickle.dump({"candidates": candidates,'scores':scores},open('results/fits.p','wb'))

for _ in range(90):
    gp = SingleTaskGP(train_X=candidates, train_Y=scores)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    logEI = LogExpectedImprovement(model=gp, best_f=scores.max().item())
    gamma, _ = optimize_acqf(
        acq_function=logEI,
        bounds=bounds,
        q=1,
        num_restarts=100,
        raw_samples=4096,
        options={"batch_limit": 8, "maxiter": 300},
    )
    candidates = torch.cat([candidates, gamma], dim=0).to(torch.double)
    vb = SMI_VB(gamma.to(torch.float32),dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
    vb.train(iterations_training)
    scores = torch.cat([scores,score(vb).reshape((1,1))], dim=0).to(torch.double)
    pickle.dump({"candidates": candidates,'scores':scores},open('results/fits.p','wb'))
    pickle.dump(vb,open('results/smi_'+str(num_model)+'.p','wb'))
    print(str(num_model)+' done')
    num_model += 1
    

vb_conventional =  pickle.load(open('results/vb_conventional.p', 'rb'))

vb_cut = pickle.load(open('results/vb_cut.p', 'rb'))

results = pickle.load(open('results/fits.p', 'rb'))
candidates = results['candidates']
scores = results['scores']

print(candidates[scores.argmax().item()])
vb_smi = pickle.load(open('results/smi_'+str(scores.argmax().item())+'.p','rb'))

color_conventional = 'tab:blue'
color_cut = 'tab:green'
color_smi = 'tab:red' 
fig_width = 8
size_labels = 8
size_titles = 10

score_smi = scores.max().item()
score_conventional = score(vb_conventional)
score_cut = score(vb_cut)
opt = [scores[:(t+1)].max().item() for t in range(100)]

fig, ax = plt.subplots(dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
ax.axhline(score_smi,color='gray')
ax.plot(np.arange(1,101),opt,color='black')
ax.grid(alpha=0.3)
ax.set_xlabel(r'number of $\gamma$ checked')
ax.set_ylabel(r'$u(\gamma)$ under $\gamma^\ast$')
ax.axhline(score_cut,color='gray')
ax.axhline(score_conventional,color='gray')
ax.text(27,score_cut+0.00001,'fully cut',color='gray')
ax.text(27,score_conventional+0.00001,'conventional',color='gray')
ax.text(27,score_smi+0.00001,'optimal SMI',color='gray')
fig.tight_layout()
plt.savefig("sim2_utility.pdf", format="pdf", bbox_inches="tight")
plt.show()


eta_smi = vb_smi.mu_eta
psi_smi = vb_smi.mu_psi
eta_conventional = vb_conventional.mu_eta
psi_conventional = vb_conventional.mu_psi
eta_cut = vb_cut.mu_eta
psi_cut = vb_cut.mu_psi


fig, axs = plt.subplots(3,block_size,dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
for j in range(int(3*block_size)):
    ax = axs.flatten(order='F')[j]
    ax.hist(y[:,j],color='gray',alpha=0.4,bins=20,density=True,label='data')
    pos = torch.linspace(min(y[:,j]),max(y[:,j]),5000)
    ax.plot(pos,marginals[j].pdf(pos),color='gray',label='true density')
    ax.plot(pos,torch.distributions.normal.Normal(eta_conventional[2*j],eta_conventional[2*j+1].exp()).log_prob(pos).exp(),color=color_conventional,alpha=0.9,label='conventional',linestyle="--")
    ax.plot(pos,torch.distributions.normal.Normal(eta_cut[2*j],eta_cut[2*j+1].exp()).log_prob(pos).exp(),color=color_cut,alpha=0.9,label='fully cut',linestyle=(0, (3, 1, 1, 1, 1, 1)))
    ax.plot(pos,torch.distributions.normal.Normal(eta_smi[2*j],eta_smi[2*j+1].exp()).log_prob(pos).exp(),color=color_smi,alpha=0.9,label='optimal SMI',linestyle='-.')
    ax.set_title('Marginal '+str(j+1))
for n_ax, ax in enumerate(axs.flatten()):
    ax.tick_params(axis='both', labelsize=7)
    ax.text(-0.1, 1.1, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
    ax.grid(alpha=0.3,linestyle='--')
handles, labels = axs.flatten()[0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0),frameon=False)
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("sim2_marginals.pdf", format="pdf", bbox_inches="tight")

psi_cut_sample,psi_smi_sample,psi_conventional_sample = [],[],[]
for draw in range(50000):
    _, psi_cut_draw, _, _ = vb_cut.single_sample()  
    psi_cut_draw[0] = torch.exp(psi_cut_draw[0])
    psi_cut_sample.append(psi_cut_draw.flatten())
    _, psi_smi_draw, _, _ = vb_smi.single_sample()  
    psi_smi_draw[0] = torch.exp(psi_smi_draw[0])
    psi_smi_sample.append(psi_smi_draw.flatten())
    _, psi_conventional_draw, _, _ = vb_conventional.single_sample()  
    psi_conventional_draw[0] = torch.exp(psi_conventional_draw[0])
    psi_conventional_sample.append(psi_conventional_draw.flatten())
psi_cut_sample = torch.stack(psi_cut_sample)
psi_conventional_sample = torch.stack(psi_conventional_sample)
psi_smi_sample = torch.stack(psi_smi_sample)

fig, axs = plt.subplots(3, block_size, dpi=800, figsize=(fig_width, ((5**.5 - 1) / 2)*fig_width))
for j in range(int(3*block_size)):
    ax = axs.flatten(order='F')[j]
    data_cut = torch.tanh(psi_cut_sample[:, j]).cpu().numpy()
    data_conventional = torch.tanh(psi_conventional_sample[:, j]).cpu().numpy()
    data_smi = torch.tanh(psi_smi_sample[:, j]).cpu().numpy()
    kde_cut = gaussian_kde(data_cut)
    kde_conventional = gaussian_kde(data_conventional)
    kde_smi = gaussian_kde(data_smi)
    
    x_min = min(data_cut.min(), data_conventional.min(), data_smi.min())
    x_max = max(data_cut.max(), data_conventional.max(), data_smi.max())
    x_range = np.linspace(x_min, x_max, 200)
    
    ax.plot(x_range, kde_cut(x_range), color=color_cut, alpha=0.9, label="cut",linestyle=(0, (3, 1, 1, 1, 1, 1)))
    ax.plot(x_range, kde_conventional(x_range), color=color_conventional, alpha=0.9, label="conventional",linestyle="--")
    ax.plot(x_range, kde_smi(x_range), color=color_smi, alpha=0.9, label="smi",linestyle='-.')
    
    ax.axvline(true_loadings[j], color="gray", label=r"true")
    ax.set_title('$\\ell_{'+str(j+1)+'}$')

for n_ax, ax in enumerate(axs.flatten()):
    ax.tick_params(axis='both', labelsize=7)
    ax.text(-0.1, 1.1, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
    ax.grid(alpha=0.3, linestyle='--')

handles, labels = axs.flatten()[0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0), frameon=False)
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("sim2_loadings.pdf", format="pdf", bbox_inches="tight")
plt.show()


print('Matrix norm')
print('conventional',torch.linalg.norm(psi_to_omega(psi_conventional)-omega_true).item())
print('cut',torch.linalg.norm(psi_to_omega(psi_cut)-omega_true).item())
print('SMI',torch.linalg.norm(psi_to_omega(psi_smi)-omega_true).item())
