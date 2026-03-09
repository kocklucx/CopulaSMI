import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula, IndependenceCopula
import string
import os
import pickle
from copula_smi import SMI_VB

def log_pdf_copula(u,psi):
    u.clamp_(1e-5,1-1e-5)
    theta = 1/(1-torch.special.expit(psi))
    xy = (-torch.log(u))
    xy_theta = xy.pow(theta)
    sum_xy_theta = torch.sum(xy_theta, axis=-1)
    sum_xy_theta_theta = sum_xy_theta ** (1.0 / theta)
    a = -sum_xy_theta_theta
    b = torch.log(sum_xy_theta_theta + theta - 1.0)
    c = (1.0 / theta - 2)*torch.log(sum_xy_theta)
    d = (theta - 1.0)*torch.sum(torch.log(xy), axis=-1)
    e = -torch.log(torch.prod(u, axis=-1))
    return (a+b+c+d+e).sum()

def log_pdf_marginals(y,theta):
    l = 0
    for j in range(len(y[0])):
        l+= torch.distributions.log_normal.LogNormal(theta[2*j],theta[2*j+1].exp().sqrt()).log_prob(y[:,j]).sum()
    return l

def cdfs_marginals(y,theta):
    mu = theta[[2*j for j in range(len(y[0]))]].flatten()
    sigma = theta[[2*j+1 for j in range(len(y[0]))]].exp().sqrt().flatten()
    cdf = torch.distributions.log_normal.LogNormal(mu,sigma).cdf(y)
    return cdf.clip(0,1)

def log_prior(psi,theta):
    p_psi =  (torch.special.expit(psi)*(1-torch.special.expit(psi))).sum()
    p_mu = torch.distributions.normal.Normal(loc=0,scale=100).log_prob(theta[[0,2]]).sum()
    p_sigma = (torch.distributions.half_normal.HalfNormal(100).log_prob(theta[[1,3]].exp())+theta[[1,3]]).sum()
    return p_psi + p_mu + p_sigma
 
joint_dist = CopulaDistribution(copula=GumbelCopula(theta=GumbelCopula().theta_from_tau(0.7)), marginals=[stats.lognorm(s=1,scale=np.exp(-1)),stats.gamma(a=2)])

grid = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
anz_runs = 100

# learn the SMI variational approximation for all 100 data sets on the 11^2 specifications of gamma. Due to the large number
# of iterations: This takes a while.  
for run in range(anz_runs):
    y = torch.from_numpy(joint_dist.rvs(1000,random_state=np.random.randint(999999)))
    pickle.dump(y,open('results_simulation/traindata_'+str(run+1)+'.p','wb'))
    for j1 in range(len(grid)):
        for j2 in range(len(grid)):
            gamma1 = grid[j1]
            gamma2 = grid[j2]
            gamma = torch.tensor([gamma1,gamma2])
            vb = SMI_VB(gamma,1,4,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
            vb.train(50000)
            pickle.dump(vb,open('results_simulation/smi_'+str(int(100*gamma1))+'_'+str(int(100*gamma2))+'_'+str(run+1)+'.p','wb'))

# calculate l1, l2, l_copula, and the utility function
ll1 = [[[] for _ in range(len(grid))] for _ in range(len(grid))]
ll2 = [[[] for _ in range(len(grid))] for _ in range(len(grid))]
llc = [[[] for _ in range(len(grid))] for _ in range(len(grid))]
utility = [[[] for _ in range(len(grid))] for _ in range(len(grid))]

for run in range(anz_runs):
    y = pickle.load(open('results_simulation/traindata_'+str(run+1)+'.p', 'rb')).numpy()
    n = len(y)
    spearman_rank = stats.kendalltau(y[:,0],y[:,1])[0]
    for j1 in range(len(grid)):
        for j2 in range(len(grid)):
            gamma1 = grid[j1]
            gamma2 = grid[j2]
            
            gamma = torch.tensor([gamma1,gamma2])
            vb = pickle.load(open('results_simulation/smi_'+str(int(100*gamma1))+'_'+str(int(100*gamma2))+'_'+str(run+1)+'.p','rb'))
            
            post_samples = []
            for _ in range(10000):
                z, psi, theta, theta_tilde = vb.single_sample()
                post_samples.append(torch.hstack([psi.flatten(),theta.flatten()]))
            post_samples = torch.vstack(post_samples)
            post_samples[:,0] = torch.special.expit(post_samples[:,0])
            post_samples[:,[2,4]] = post_samples[:,[2,4]].exp()
            tau,mu1,sigma1,mu2,sigma2 = post_samples[:,0],post_samples[:,1],post_samples[:,2],post_samples[:,3],post_samples[:,4]
            
            ll1[j1][j2].append((.5*((mu1+1).pow(2)+sigma1.pow(2)-1)-sigma1.log()).mean().item())
            ll2[j1][j2].append((-2*mu2-.5*torch.log(2*torch.pi*torch.e*sigma2.pow(2))+torch.exp(mu2+sigma2.pow(2)/2)).mean().item())
            llc[j1][j2].append((0.7-tau).pow(2).mean().item())
            u = []
            for j in range(10000):
                distr3_hat = CopulaDistribution(copula=IndependenceCopula(), marginals=[stats.lognorm(s=sigma1[j].numpy(),scale=np.exp(mu1[j].numpy())),stats.lognorm(s=sigma2[j].numpy(),scale=np.exp(mu2[j].numpy()))])
                u.append(distr3_hat.logpdf(y).mean())
            utility[j1][j2].append(np.mean(u))

# generate figure
fig_width = 8
size_labels = 10
size_titles = 10

fig,axs = plt.subplots(2,2,dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
mat = axs[0,0].imshow(np.mean(ll1,axis=2)[::-1],cmap='gray_r')
cbar = plt.colorbar(mat,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
axs[0,0].set_title('Marginal 1')

mat = axs[0,1].imshow(np.mean(ll2,axis=2)[::-1],cmap='gray_r')
cbar = plt.colorbar(mat,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
axs[0,1].set_title('Marginal 2')

mat = axs[1,0].imshow(np.mean(llc,axis=2)[::-1],cmap='gray_r')
cbar = plt.colorbar(mat,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
axs[1,0].set_title('Copula')

mat = axs[1,1].imshow(-np.mean(utility,axis=2)[::-1],cmap='gray_r')
cbar = plt.colorbar(mat,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
axs[1,1].set_title('neg. Utility')

for n_ax, ax in enumerate(axs.flatten()):
    ax.tick_params(axis='both', labelsize=7)
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel(r'$\gamma_2$')
    ax.set_ylabel(r'$\gamma_1$',rotation='horizontal', ha='right')
    ax.set_xticks(np.arange(len(grid)),grid)
    ax.set_yticks(np.arange(len(grid)),grid[::-1])
    ax.text(-0.2, 1.05, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
fig.tight_layout()
fig.subplots_adjust(wspace=-0.3)
plt.show()