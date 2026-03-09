from scipy.stats import skewnorm, kendalltau
from torch.autograd import Function
import numpy as np
import pandas as pd
import torch
import math
import pickle
import matplotlib.pyplot as plt
import string
from matplotlib.ticker import  MultipleLocator
from collections import OrderedDict
from copula_smi import SMI_VB

def normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def psi_to_paras(psi):
    psi = psi.flatten()
    d_half = d // 2
    n_spatial = d_half * (d_half - 1) // 2
    params_spatial = psi[:n_spatial]
    params_temp    = psi[n_spatial:-d_half]
    params_skew    = psi[-d_half:]
    params_spatial = 2*normal_cdf(params_spatial) - 1
    params_temp    = 2*normal_cdf(params_temp) - 1
    dtype = psi.dtype
    pac_eye = torch.eye(d,dtype=dtype)
    tri = torch.triu_indices(d_half, d_half, offset=1)
    i0, j0 = tri[0], tri[1]
    pac_flat = pac_eye.reshape(-1)
    lin_idx_1 = i0 * d + j0
    pac_flat = pac_flat.scatter(0, lin_idx_1, params_spatial)
    lin_idx_2 = (i0 + d_half) * d + (j0 + d_half)
    pac_flat = pac_flat.scatter(0, lin_idx_2, params_spatial)
    pac = pac_flat.reshape((d,d))
    temp_block = params_temp.reshape(d_half, d_half)
    add_temp = torch.zeros_like(pac)
    add_temp[:d_half, d_half:] = temp_block
    pac = pac + add_temp  
    omega = torch.eye(d, dtype=dtype)
    for k in range(1, d):
        omega_new = omega.clone()
        for j in range(d - k):
            if k < 2:
                val = pac[j, j+k]
                omega_new[j, j+k] = val
                omega_new[j+k, j] = val
            else:
                r1 = omega[j, j+1:j+k]
                r2 = omega[j+1:j+k, j+1:j+k]
                r3 = omega[j+k, j+1:j+k]
                s1 = torch.linalg.solve(r2, r1)
                s3 = torch.linalg.solve(r2, r3)
                d2 = (1 - r1 @ s1) * (1 - r3 @ s3)
                d2 = torch.clamp(d2, min=0.0)
                val = r1 @ s3 + d2.sqrt() * pac[j, j+k]
                omega_new[j, j+k] = val
                omega_new[j+k, j] = val
        omega = omega_new 
    omega_spatial  = omega[:d_half, :d_half]
    omega_temporal = omega[:d_half, d_half:]
    alpha1 = params_skew
    M = omega_spatial - omega_temporal
    alpha2 = torch.linalg.solve(M, (omega_spatial - omega_temporal.T) @ alpha1)
    alpha  = torch.hstack([alpha1, alpha2])
    return omega, alpha

def marginal_distribution_pdf(eta,j,pos):
    mu = eta[(4*j)%(2*d)]
    sigma = eta[(4*j+1)%(2*d)].exp()
    skew = eta[(4*j+2)%(2*d)]
    kurt = eta[(4*j+3)%(2*d)].exp()
    y_stand = (torch.tensor(pos)-mu)/sigma
    l = -sigma.log()- 1/2*torch.log(torch.tensor(2*torch.pi))+(kurt*torch.cosh(skew+kurt*torch.asinh(y_stand))).log()-1/2*(1+y_stand.pow(2)).log()-1/2*(torch.sinh(skew+kurt*torch.asinh(y_stand))).pow(2)
    return l.exp().numpy()

def sample_from_copula(psi,n=10000):
    omega, alpha = psi_to_paras(psi)
    delta = ((1+alpha@omega@alpha).pow(-.5)*omega@alpha).unsqueeze(-1)
    sigma = torch.vstack([torch.cat([omega,delta],dim=1),torch.cat([delta.T,torch.ones((1,1))],dim=1)])
    sample = torch.distributions.MultivariateNormal(torch.zeros(d+1),sigma).sample((n,))
    z = sample[sample[:,-1]>0,:-1]
    u = skewnorm.cdf(z, ((1-delta**2).pow(-.5)*delta).flatten())
    u = torch.tensor(u).to(torch.float32)
    return u

def score(vb):
    num_mc_samples = 10000
    lag_idx = int(d/2)
    result = []
    for _ in range(num_mc_samples):
        z, psi, eta, eta_tilde = vb.single_sample()
        mu = eta[[(4*j)%(2*d) for j in range(lag_idx)]].flatten()
        sigma = eta[[(4*j+1)%(2*d) for j in range(lag_idx)]].exp().flatten()
        skew = eta[[(4*j+2)%(2*d) for j in range(lag_idx)]].flatten()
        kurt = eta[[(4*j+3)%(2*d) for j in range(lag_idx)]].exp().flatten()
        y_stand = (y[:,:lag_idx]-mu)/sigma
        l = -sigma.log()-1/2*torch.log(torch.tensor(2*torch.pi))+(kurt*torch.cosh(skew+kurt*torch.asinh(y_stand))).log()-1/2*(1+y_stand.pow(2)).log()-1/2*(torch.sinh(skew+kurt*torch.asinh(y_stand))).pow(2)
        result.append(l)
    result = torch.stack(result)
    return result.sum(axis=2).mean(axis=0).sum()

def major_minor_asymmetry(u_sample,i,j,quantile=0.05):
    ll = np.mean(u_sample[u_sample[:,i]<=quantile,j]<=quantile) #lower left
    ur = np.mean(u_sample[u_sample[:,i]>1-quantile,j]>1-quantile) #upper right
    lr = np.mean(u_sample[u_sample[:,i]>1-quantile,j]<=quantile) # lower right
    ul = np.mean(u_sample[u_sample[:,i]<=quantile,j]>1-quantile)
    major = ur-ll
    minor = ul-lr
    return major,minor

def copula_parameters(psi,i,j):
    omega, alpha = psi_to_paras(psi)
    delta = ((1+alpha@omega@alpha).pow(-.5)*omega@alpha).unsqueeze(-1)
    sigma = torch.vstack([torch.cat([omega,delta],dim=1),torch.cat([delta.T,torch.ones((1,1))],dim=1)])
    sigma = sigma[[[i,i,i],[j,j,j],[-1,-1,-1]],[[i,j,-1],[i,j,-1],[i,j,-1]]]
    omega = sigma[[[0,0],[1,1]],[[0,1],[0,1]]]
    delta = sigma[2,[0,1]]
    alpha = (1-delta@torch.linalg.solve(omega,delta)).pow(-.5)*torch.linalg.solve(omega,delta)
    return sigma[0,1].item(),alpha

# summarize the results for the financial application and generate the plots as shown in the manuscript. 
# run financial_application.py first!

#### load data
df = pd.read_csv('yields.csv',index_col='Date')
names = ['VIX', 'AAA', 'BBB', 'VIXlag', 'AAAlag', 'BBBlag']#df.columns

y = torch.tensor(np.asarray(df)).to(torch.float32)
n=len(y)
d = len(y[0])
d_half = int(d/2)
dim_psi,dim_eta = int(d_half*(d_half-1)/2+d_half**2+d_half),2*d

vb_conventional = pickle.load(open('results_application/vb_conventional.p','rb'))  
eta_conventional = vb_conventional.mu_eta
psi_conventional = vb_conventional.mu_psi

vb_cut = pickle.load(open('results_application/vb_cut.p','rb'))  
eta_cut = vb_cut.mu_eta
psi_cut = vb_cut.mu_psi

results = pickle.load(open('results_application/fits.p','rb'))
candidates = results['candidates']
scores = results['scores']

vb_smi = pickle.load(open('results_application/smi_'+str(scores.argmax().item())+'.p','rb'))
eta_smi = vb_smi.mu_eta
psi_smi = vb_smi.mu_psi

u_conventional = sample_from_copula(psi_conventional,500000)
u_cut = sample_from_copula(psi_cut,500000)
u_smi = sample_from_copula(psi_smi,500000)

color_conventional = 'tab:blue'
color_cut = 'tab:green'
color_smi = 'tab:red' 
fig_width = 8
size_labels = 8
size_titles = 10

u_obs = np.asarray([[(y[:,d]<=y[l,d]).sum() for l in range(len(y))] for d in range(len(y[0]))]).T/(len(y)+1)
fig, axs = plt.subplots(5,3,dpi=800,figsize=(fig_width,2*((5**.5 - 1) / 2)*fig_width),sharex=True,sharey=True)
for h in range(5):
    j1,j2 = [(0,1),(0,2),(1,2),(3,1),(3,2)][h]#np.random.randint(0,d,2)#
    axs[h,0].set_ylabel(names[j1]+' - '+names[j2])
    ll_obs,ur_obs,lr_obs,ul_obs = [0],[0],[0],[0]
    for u in np.linspace(0,.5,50)[1:]:
        ll_obs.append(np.mean(u_obs[u_obs[:,j1]<=u,j2]<=u)) #lower left
        ur_obs.append(np.mean(u_obs[u_obs[:,j1]>1-u,j2]>1-u)) #upper right
        lr_obs.append(np.mean(u_obs[u_obs[:,j1]>1-u,j2]<=u)) # lower right
        ul_obs.append(np.mean(u_obs[u_obs[:,j1]<=u,j2]>1-u)) # upper left
    for method in range(3):
        ax = axs[h,method]
        u_sample = [u_conventional.numpy(),u_cut.numpy(),u_smi.numpy()][method]
        ll,ur,lr,ul = [0],[0],[0],[0]
        for u in np.linspace(0,.5,50)[1:]:
            ll.append(np.mean(u_sample[u_sample[:,j1]<=u,j2]<=u)) #lower left
            ur.append(np.mean(u_sample[u_sample[:,j1]>1-u,j2]>1-u)) #upper right
            lr.append(np.mean(u_sample[u_sample[:,j1]>1-u,j2]<=u)) # lower right
            ul.append(np.mean(u_sample[u_sample[:,j1]<=u,j2]>1-u)) # upper left
        ax.plot(np.linspace(0,.5,50),ll,label='lower left',color='tab:blue')
        ax.plot(1-np.linspace(0,.5,50),ur,label='upper right',color='tab:orange')
        ax.plot(np.linspace(0,.5,50),lr,label='lower right',color='tab:green')
        ax.plot(1-np.linspace(0,.5,50),ul,label='upper left',color='tab:red')
        ax.plot(np.linspace(0,.5,50),ll_obs,color='gray',alpha=0.6)
        ax.plot(1-np.linspace(0,.5,50),ur_obs,color='gray',alpha=0.6)
        ax.plot(np.linspace(0,.5,50),lr_obs,color='gray',alpha=0.6)
        ax.plot(1-np.linspace(0,.5,50),ul_obs,color='gray',alpha=0.6)
        #ax.legend(fontsize='x-small')
axs[0,0].set_title('Conventional')
axs[0,1].set_title('Fully cut')
axs[0,2].set_title('Optimal SMI')
for n_ax, ax in enumerate(axs.flatten()):
    ax.tick_params(axis='both', labelsize=7)
    ax.text(0.00, 1.05, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
    ax.grid(alpha=0.3,linestyle='--')
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(alpha=0.3,linestyle=':',which='minor')
h, l = ax.get_legend_handles_labels()
fig.legend(h, l,loc="lower center",ncol=4,frameon=False)
fig.tight_layout()
plt.subplots_adjust(bottom=0.05) 
plt.show()

fig, axs = plt.subplots(1,3,dpi=800,figsize=(fig_width,.5*((5**.5 - 1) / 2)*fig_width))
for j in range(3):
    ax = axs.flatten()[j]
    ax.hist(y[:,j],color='gray',alpha=0.4,bins=40,density=True,label='data')
    pos = np.linspace(min(y[:,j]),max(y[:,j]),5000)
    ax.plot(pos,marginal_distribution_pdf(vb_conventional.mu_eta,j,pos),color=color_conventional,alpha=0.9,label='conventional',linestyle="--")
    ax.plot(pos,marginal_distribution_pdf(vb_cut.mu_eta,j,pos),color=color_cut,alpha=0.9,label='fully cut',linestyle=(0, (3, 1, 1, 1, 1, 1)))
    ax.plot(pos,marginal_distribution_pdf(vb_smi.mu_eta,j,pos),color=color_smi,alpha=0.9,label='optimal SMI',linestyle='-.')
    ax.set_title(names[j],size=size_titles)
    ax.legend(fontsize='x-small')
for n_ax, ax in enumerate(axs.flatten()):
    ax.tick_params(axis='both', labelsize=7)
    ax.text(0.05, 1.05, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
    ax.grid(alpha=0.3,linestyle='--')
fig.tight_layout()
plt.show()


opt = [scores[:(t+1)].max().item() for t in range(100)]
fig, ax = plt.subplots(dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
ax.axhline(score(vb_smi),color='gray')
ax.plot(np.arange(1,101),opt,color='black')
ax.grid(alpha=0.3)
ax.set_xlabel(r'number of $\gamma$ checked')
ax.set_ylabel(r'utility under $\gamma^\ast$')
ax.axhline(score(vb_cut),color='gray')
ax.axhline(score(vb_conventional),color='gray')
ax.text(27,1.001*score(vb_cut),'fully cut',color='gray')
ax.text(27,1.001*score(vb_conventional),'conventional',color='gray')
ax.text(27,1.001*score(vb_smi),'optimal SMI',color='gray')
fig.tight_layout()
plt.show()


print(r"\begin{tabular}{ccccccc}&  &  & \multicolumn{2}{c}{10 \% Quantile asymmetry} & \multicolumn{2}{c}{Parameters} \\& Pair & Kendall's $\tau$ & $\Delta_\text{major}(0.1)$ & $\Delta_\text{minor}(0.1)$ & $\rho$ & $\alpha$ \\ \hline")
for i in range(int(d/2)):
    for j in range(i+1,d):
        # conventional
        tau = np.round(kendalltau(u_conventional[:,i],u_conventional[:,j])[0],4)
        major,minor = major_minor_asymmetry(u_conventional.numpy(),i,j,quantile=0.1)
        major, minor = np.round(major,4), np.round(minor,4)
        rho,alpha = copula_parameters(psi_conventional,i,j)
        rho, a1,a2 = np.round(rho,4),np.round(alpha[0].item(),4),np.round(alpha[1].item(),4)
        print(r'conventional & \multirow{3}{*}{'+names[i]+'--'+names[j]+'} & ',tau,'&',major,'&',minor,'&',rho,'& $(',a1,',',a2,r')^\top$',r'\\')
        # cut
        tau = np.round(kendalltau(u_cut[:,i],u_cut[:,j])[0],4)
        major,minor = major_minor_asymmetry(u_cut.numpy(),i,j,quantile=0.1)
        major, minor = np.round(major,4), np.round(minor,4)
        rho,alpha = copula_parameters(psi_cut,i,j)
        rho, a1,a2 = np.round(rho,4),np.round(alpha[0].item(),4),np.round(alpha[1].item(),4)
        print('cut & & ',tau,'&',major,'&',minor,'&',rho,'& $(',a1,',',a2,r')^\top$',r'\\')
        #smi
        tau = np.round(kendalltau(u_smi[:,i],u_smi[:,j])[0],4)
        major,minor = major_minor_asymmetry(u_smi.numpy(),i,j,quantile=0.1)
        major, minor = np.round(major,4), np.round(minor,4)
        rho,alpha = copula_parameters(psi_smi,i,j)
        rho, a1,a2 = np.round(rho,4),np.round(alpha[0].item(),4),np.round(alpha[1].item(),4)
        print('optimal SMI & & ',tau,'&',major,'&',minor,'&',rho,'& $(',a1,',',a2,r')^\top$',r'\\')
        print(r'\hline')
print(r"\end{tabular}")

u_obs = np.asarray([[(y[:,d]<=y[l,d]).sum() for l in range(len(y))] for d in range(len(y[0]))]).T/(len(y)+1)
u_sample = u_cut.numpy()
fig, axs = plt.subplots(3,4,dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
j = 0
for j1 in range(int(d/2)):
    for j2 in range(j1+1,d):
        ax = axs.flatten()[j]
        ll_obs,ur_obs,lr_obs,ul_obs = [0],[0],[0],[0]
        for u in np.linspace(0,.5,50)[1:]:
            ll_obs.append(np.mean(u_obs[u_obs[:,j1]<=u,j2]<=u)) #lower left
            ur_obs.append(np.mean(u_obs[u_obs[:,j1]>1-u,j2]>1-u)) #upper right
            lr_obs.append(np.mean(u_obs[u_obs[:,j1]>1-u,j2]<=u)) # lower right
            ul_obs.append(np.mean(u_obs[u_obs[:,j1]<=u,j2]>1-u)) # upper left
        ll,ur,lr,ul = [0],[0],[0],[0]
        for u in np.linspace(0,.5,50)[1:]:
            ll.append(np.mean(u_sample[u_sample[:,j1]<=u,j2]<=u)) #lower left
            ur.append(np.mean(u_sample[u_sample[:,j1]>1-u,j2]>1-u)) #upper right
            lr.append(np.mean(u_sample[u_sample[:,j1]>1-u,j2]<=u)) # lower right
            ul.append(np.mean(u_sample[u_sample[:,j1]<=u,j2]>1-u)) # upper left
        ax.plot(np.linspace(0,.5,50),ll,label='lower left',color='tab:blue')
        ax.plot(1-np.linspace(0,.5,50),ur,label='upper right',color='tab:orange')
        ax.plot(np.linspace(0,.5,50),lr,label='lower right',color='tab:green')
        ax.plot(1-np.linspace(0,.5,50),ul,label='upper left',color='tab:red')
        ax.plot(np.linspace(0,.5,50),ll_obs,color='gray',alpha=0.6,label='empirical')
        ax.plot(1-np.linspace(0,.5,50),ur_obs,color='gray',alpha=0.6)
        ax.plot(np.linspace(0,.5,50),lr_obs,color='gray',alpha=0.6)
        ax.plot(1-np.linspace(0,.5,50),ul_obs,color='gray',alpha=0.6)
        ax.tick_params(axis='both', labelsize=7)
        ax.text(-0.05, 1.1, string.ascii_uppercase[j], transform=ax.transAxes, weight='bold')
        ax.grid(alpha=0.3,linestyle='--')
        ax.set_title(r''+names[j1]+'-'+names[j2])
        j+=1
handles, labels = axs.flatten()[0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()

u_obs = np.asarray([[(y[:,d]<=y[l,d]).sum() for l in range(len(y))] for d in range(len(y[0]))]).T/(len(y)+1)
u_sample = u_conventional.numpy()
fig, axs = plt.subplots(3,4,dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
j = 0
for j1 in range(int(d/2)):
    for j2 in range(j1+1,d):
        ax = axs.flatten()[j]
        ll_obs,ur_obs,lr_obs,ul_obs = [0],[0],[0],[0]
        for u in np.linspace(0,.5,50)[1:]:
            ll_obs.append(np.mean(u_obs[u_obs[:,j1]<=u,j2]<=u)) #lower left
            ur_obs.append(np.mean(u_obs[u_obs[:,j1]>1-u,j2]>1-u)) #upper right
            lr_obs.append(np.mean(u_obs[u_obs[:,j1]>1-u,j2]<=u)) # lower right
            ul_obs.append(np.mean(u_obs[u_obs[:,j1]<=u,j2]>1-u)) # upper left
        ll,ur,lr,ul = [0],[0],[0],[0]
        for u in np.linspace(0,.5,50)[1:]:
            ll.append(np.mean(u_sample[u_sample[:,j1]<=u,j2]<=u)) #lower left
            ur.append(np.mean(u_sample[u_sample[:,j1]>1-u,j2]>1-u)) #upper right
            lr.append(np.mean(u_sample[u_sample[:,j1]>1-u,j2]<=u)) # lower right
            ul.append(np.mean(u_sample[u_sample[:,j1]<=u,j2]>1-u)) # upper left
        ax.plot(np.linspace(0,.5,50),ll,label='lower left',color='tab:blue')
        ax.plot(1-np.linspace(0,.5,50),ur,label='upper right',color='tab:orange')
        ax.plot(np.linspace(0,.5,50),lr,label='lower right',color='tab:green')
        ax.plot(1-np.linspace(0,.5,50),ul,label='upper left',color='tab:red')
        ax.plot(np.linspace(0,.5,50),ll_obs,color='gray',alpha=0.6,label='empirical')
        ax.plot(1-np.linspace(0,.5,50),ur_obs,color='gray',alpha=0.6)
        ax.plot(np.linspace(0,.5,50),lr_obs,color='gray',alpha=0.6)
        ax.plot(1-np.linspace(0,.5,50),ul_obs,color='gray',alpha=0.6)
        ax.tick_params(axis='both', labelsize=7)
        ax.text(-0.05, 1.1, string.ascii_uppercase[j], transform=ax.transAxes, weight='bold')
        ax.grid(alpha=0.3,linestyle='--')
        ax.set_title(r''+names[j1]+'-'+names[j2])
        j+=1
handles, labels = axs.flatten()[0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()

u_obs = np.asarray([[(y[:,d]<=y[l,d]).sum() for l in range(len(y))] for d in range(len(y[0]))]).T/(len(y)+1)
u_sample = u_smi.numpy()
fig, axs = plt.subplots(3,4,dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
j = 0
for j1 in range(int(d/2)):
    for j2 in range(j1+1,d):
        ax = axs.flatten()[j]
        ll_obs,ur_obs,lr_obs,ul_obs = [0],[0],[0],[0]
        for u in np.linspace(0,.5,50)[1:]:
            ll_obs.append(np.mean(u_obs[u_obs[:,j1]<=u,j2]<=u)) #lower left
            ur_obs.append(np.mean(u_obs[u_obs[:,j1]>1-u,j2]>1-u)) #upper right
            lr_obs.append(np.mean(u_obs[u_obs[:,j1]>1-u,j2]<=u)) # lower right
            ul_obs.append(np.mean(u_obs[u_obs[:,j1]<=u,j2]>1-u)) # upper left
        ll,ur,lr,ul = [0],[0],[0],[0]
        for u in np.linspace(0,.5,50)[1:]:
            ll.append(np.mean(u_sample[u_sample[:,j1]<=u,j2]<=u)) #lower left
            ur.append(np.mean(u_sample[u_sample[:,j1]>1-u,j2]>1-u)) #upper right
            lr.append(np.mean(u_sample[u_sample[:,j1]>1-u,j2]<=u)) # lower right
            ul.append(np.mean(u_sample[u_sample[:,j1]<=u,j2]>1-u)) # upper left
        ax.plot(np.linspace(0,.5,50),ll,label='lower left',color='tab:blue')
        ax.plot(1-np.linspace(0,.5,50),ur,label='upper right',color='tab:orange')
        ax.plot(np.linspace(0,.5,50),lr,label='lower right',color='tab:green')
        ax.plot(1-np.linspace(0,.5,50),ul,label='upper left',color='tab:red')
        ax.plot(np.linspace(0,.5,50),ll_obs,color='gray',alpha=0.6,label='empirical')
        ax.plot(1-np.linspace(0,.5,50),ur_obs,color='gray',alpha=0.6)
        ax.plot(np.linspace(0,.5,50),lr_obs,color='gray',alpha=0.6)
        ax.plot(1-np.linspace(0,.5,50),ul_obs,color='gray',alpha=0.6)
        ax.tick_params(axis='both', labelsize=7)
        ax.text(-0.05, 1.1, string.ascii_uppercase[j], transform=ax.transAxes, weight='bold')
        ax.grid(alpha=0.3,linestyle='--')
        ax.set_title(r''+names[j1]+'-'+names[j2])
        j+=1
handles, labels = axs.flatten()[0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()

major_cut = np.zeros((d,d))
minor_cut = np.zeros((d,d))
major_smi = np.zeros((d,d))
minor_smi = np.zeros((d,d))

for j in range(d):
    for i in range(d):
        major, minor = major_minor_asymmetry(u_cut.numpy(),i,j,quantile=0.1)
        major_cut[i,j] = major
        minor_cut[i,j] = minor
        major, minor = major_minor_asymmetry(u_smi.numpy(),i,j,quantile=0.1)
        major_smi[i,j] = major
        minor_smi[i,j] = minor

fig_width = 6
size_labels = 8
size_titles = 10
vmax = 0.25
vmin = -vmax
fig,axs = plt.subplots(2,2,dpi=800,figsize=(fig_width,fig_width))
#
mat = major_cut
masked = np.tril(mat)
data = np.where(np.tri(mat.shape[0], k=-1, dtype=bool), mat, np.nan)
cax = axs[0,0].imshow(data[1:,:-1], vmin=vmin, vmax=vmax, cmap='coolwarm')
axs[0,0].set_xticks(range(mat.shape[0]-1))
axs[0,0].set_yticks(range(mat.shape[0]-1))
axs[0,0].set_title(r'$\Delta_{Major}(0.1)$ ─ fully cut')
#
mat = minor_cut
masked = np.tril(mat)
data = np.where(np.tri(mat.shape[0], k=-1, dtype=bool), mat, np.nan)
cax = axs[0,1].imshow(data[1:,:-1], vmin=vmin, vmax=vmax, cmap='coolwarm')
axs[0,1].set_xticks(range(mat.shape[0]-1))
axs[0,1].set_yticks(range(mat.shape[0]-1))
axs[0,1].set_title(r'$\Delta_{Minor}(0.1)$ ─ fully cut')
#
mat = major_smi
masked = np.tril(mat)
data = np.where(np.tri(mat.shape[0], k=-1, dtype=bool), mat, np.nan)
cax = axs[1,0].imshow(data[1:,:-1], vmin=vmin, vmax=vmax, cmap='coolwarm')
axs[1,0].set_xticks(range(mat.shape[0]-1))
axs[1,0].set_yticks(range(mat.shape[0]-1))
axs[1,0].set_title(r'$\Delta_{Major}(0.1)$ ─ optimal SMI')
#
mat = minor_smi
masked = np.tril(mat)
data = np.where(np.tri(mat.shape[0], k=-1, dtype=bool), mat, np.nan)
cax = axs[1,1].imshow(data[1:,:-1], vmin=vmin, vmax=vmax, cmap='coolwarm')
axs[1,1].set_xticks(range(mat.shape[0]-1))
axs[1,1].set_yticks(range(mat.shape[0]-1))
axs[1,1].set_title(r'$\Delta_{Minor}(0.1)$ ─ optimal SMI')
for n_ax, ax in enumerate(axs.flatten()):
    ax.tick_params(axis='both', labelsize=7)
    ax.text(-0.2, 1.05, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
    ax.set_xticklabels(names[:-1],fontsize=size_labels)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_yticklabels(names[1:],fontsize=size_labels)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.subplots_adjust(left=0.06, right=0.82, wspace=0.5,hspace=0.5)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=size_labels)
cbar.set_label("pairwise asymmetry",fontsize=size_labels,labelpad=5)
cbar.ax.yaxis.set_label_position('left')
plt.show()