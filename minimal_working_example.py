from copula_smi import SMI_VB
import torch
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# define function for the statistical model. 
# log_pdf_copula evaluates the log copula likelihood
# log_pdf_marginals evaluates the log likelihood for the marginals
# cdfs_marginals returns CDF values for y evaluated at eta
# log_prior is the log-pdf of the prior for the model parameters
# Here, we consider a Gumbel copula with log-normal marginals

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

# generate data. Here, the data comes from a Gumbel copula model with log-normal and gamma marginals
joint_dist = CopulaDistribution(copula=GumbelCopula(theta=GumbelCopula().theta_from_tau(0.7)), marginals=[stats.lognorm(s=1,scale=np.exp(-1)),stats.gamma(a=2)])
y = torch.from_numpy(joint_dist.rvs(1000,random_state=2026))

# specify gamma and train the variational approximation
gamma = torch.tensor([1.0,0.0])
vb = SMI_VB(gamma,1,4,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
vb.train(50000)

# now the results can be used for further analysis. For example:
z, psi, theta, theta_tilde = vb.single_sample() # is a sample from the variational posterior
psi_hat = vb.mu_psi # mean for psi
eta_hat = vb.mu_eta # mean for eta

# estimated Kendall's tau under the posterior mean (should be close to 0.7):
tau_hat = GumbelCopula(theta=1/(1-torch.special.expit(psi_hat)).numpy().item()).tau()