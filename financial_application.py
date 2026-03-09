from scipy.stats import skewnorm
from torch.autograd import Function
import numpy as np
import pandas as pd
import torch
import math
import pickle
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
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

def skew_normal_logpdf(z,omega,alpha):
    d = len(z[0])
    l = (torch.log(torch.tensor([2.]))+torch.distributions.MultivariateNormal(loc=torch.zeros(d),covariance_matrix=omega).log_prob(z)+torch.distributions.Normal(0,1).cdf(z@alpha).clip(1e-5,1-1e-5).log())
    delta = (1+alpha@omega@alpha).pow(-.5)*omega@alpha
    alpha_j = (1-delta**2).pow(-.5)*delta
    l += (-torch.log(torch.tensor([2.]))-torch.distributions.Normal(0,1).log_prob(z)-torch.distributions.Normal(0,1).cdf(z*alpha_j).log()).sum(axis=1)
    return l
    
def u_to_z(u,omega,alpha):
    delta = (1+alpha@omega@alpha).pow(-.5)*omega@alpha
    alpha = (1-delta**2).pow(-.5)*delta
    return skewnorm_ppf(alpha,u)

def log_pdf_copula(u,psi):
    u.clamp_(1e-5,1-1e-5)
    omega, alpha = psi_to_paras(psi)
    z = u_to_z(u,omega,alpha)
    return skew_normal_logpdf(z,omega,alpha).sum()

class SkewNormPPF(Function):
    @staticmethod
    def forward(ctx, a, u):
        a_np = a.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        x = skewnorm.ppf(u_np, a_np)
        x_tensor = torch.from_numpy(x).to(u.device).to(u.dtype)
        ctx.save_for_backward(a, u, x_tensor)
        return x_tensor

    @staticmethod
    def backward(ctx, grad_output):
        a, u, x = ctx.saved_tensors
        a_np = a.detach().cpu().numpy() 
        u_np = u.detach().cpu().numpy()  
        x_np = x.detach().cpu().numpy()   
        pdf = np.clip(skewnorm.pdf(x_np, a_np),a_min=1e-5,a_max=None)
        grad_u_np = 1.0 / pdf           
        eps = 1e-5
        grad_a_np = np.zeros_like(pdf)     
        for j in range(a_np.shape[0]):
            a_plus = np.array(a_np, copy=True)
            a_plus[j] += eps
            a_minus = np.array(a_np, copy=True)
            a_minus[j] -= eps
            cdf_plus = skewnorm.cdf(x_np[:, j], a_plus[j])    
            cdf_minus = skewnorm.cdf(x_np[:, j], a_minus[j])  
            dFda = (cdf_plus - cdf_minus) / (2 * eps)  
            grad_a_np[:,j] = - dFda / pdf[:, j]
        grad_a = torch.from_numpy(grad_a_np).to(a.device).to(a.dtype)  
        grad_u = torch.from_numpy(grad_u_np).to(u.device).to(u.dtype) 
        grad_a_out = (grad_output * grad_a).sum().unsqueeze(0) if a.shape == torch.Size([]) else grad_output * grad_a
        grad_u_out = grad_output * grad_u
        return grad_a_out, grad_u_out

def skewnorm_ppf(a, u):
    return SkewNormPPF.apply(a, u)

def log_pdf_marginals(y,eta):
    mu = eta[[(4*j)%(2*d) for j in range(d)]].flatten()
    sigma = eta[[(4*j+1)%(2*d) for j in range(d)]].exp().flatten()
    skew = eta[[(4*j+2)%(2*d) for j in range(d)]].flatten()
    kurt = eta[[(4*j+3)%(2*d) for j in range(d)]].exp().flatten()
    y_stand = (y-mu)/sigma
    l = -sigma.log()-1/2*torch.log(torch.tensor(2*torch.pi))+(kurt*torch.cosh(skew+kurt*torch.asinh(y_stand))).log()-1/2*(1+y_stand.pow(2)).log()-1/2*(torch.sinh(skew+kurt*torch.asinh(y_stand))).pow(2)
    return l.sum()

def cdfs_marginals(y,eta):
    mu = eta[[(4*j)%(2*d) for j in range(d)]].flatten()
    sigma = eta[[(4*j+1)%(2*d) for j in range(d)]].exp().flatten()
    skew = eta[[(4*j+2)%(2*d) for j in range(d)]].flatten()
    kurt = eta[[(4*j+3)%(2*d) for j in range(d)]].exp().flatten()
    y_stand = (y-mu)/sigma
    cdf = torch.distributions.Normal(0,1).cdf(torch.sinh(skew+kurt*torch.asinh(y_stand))).clip(1e-5,1-1e-5)
    return cdf

def log_prior(psi,eta):
    # psi
    psi = psi.flatten()
    p_psi = (torch.distributions.normal.Normal(loc=0,scale=1).log_prob(psi)).sum()
    # eta
    mu = eta[[(4*j)%(2*d) for j in range(d)]].flatten()
    sigma = eta[[(4*j+1)%(2*d) for j in range(d)]].exp().flatten()
    skew = eta[[(4*j+2)%(2*d) for j in range(d)]].flatten()
    kurt = eta[[(4*j+3)%(2*d) for j in range(d)]].exp().flatten()
    p_mu = torch.distributions.normal.Normal(loc=0,scale=10).log_prob(mu).sum()
    p_skew = torch.distributions.normal.Normal(loc=0,scale=10).log_prob(skew).sum()
    p_sigma = (torch.distributions.half_normal.HalfNormal(10).log_prob(sigma)+sigma.log()).sum() 
    p_kurt = (torch.distributions.half_normal.HalfNormal(10).log_prob(kurt)+kurt.log()).sum() 
    p_eta = p_mu+p_sigma+p_skew+p_kurt
    return p_psi + p_eta

def marginal_distribution_pdf(eta,j,pos):
    mu = eta[(4*j)%(2*d)]
    sigma = eta[(4*j+1)%(2*d)].exp()
    skew = eta[(4*j+2)%(2*d)]
    kurt = eta[(4*j+3)%(2*d)].exp()
    y_stand = (torch.tensor(pos)-mu)/sigma
    l = -sigma.log()- 1/2*torch.log(torch.tensor(2*torch.pi))+(kurt*torch.cosh(skew+kurt*torch.asinh(y_stand))).log()-1/2*(1+y_stand.pow(2)).log()-1/2*(torch.sinh(skew+kurt*torch.asinh(y_stand))).pow(2)
    return l.exp().numpy()

def score(vb):
    # utility function
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

#### load data
df = pd.read_csv('yields.csv',index_col='Date')

iterations_training = 100000

y = torch.tensor(np.asarray(df)).to(torch.float32)
n=len(y)
d = len(y[0])
d_half = int(d/2)
dim_psi,dim_eta = int(d_half*(d_half-1)/2+d_half**2+d_half),2*d

# fit cut and uncut model
vb = SMI_VB(torch.ones(d),dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
vb.train(iterations_training)
pickle.dump(vb,open('results_application/vb_conventional.p','wb'))

vb = SMI_VB(torch.zeros(d),dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
vb.train(iterations_training)
pickle.dump(vb,open('results_application/vb_cut.p','wb'))

# BO for optimal SMI
bounds = torch.stack([torch.zeros(int(d/2)), torch.ones(int(d/2))]).to(torch.double)

num_model = 0
candidates = []
scores = []
for _ in range(2):
    gamma = torch.rand(3)
    vb = SMI_VB(torch.hstack([gamma,gamma]),dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
    vb.train(iterations_training)
    pickle.dump(vb,open('results/smi_'+str(num_model)+'.p','wb'))
    scores.append(score(vb))
    candidates.append(gamma)
    num_model += 1
    
candidates = torch.vstack(candidates).to(torch.double)
scores = torch.vstack(scores).to(torch.double)
pickle.dump({"candidates": candidates,'scores':scores},open('results_application/fits.p','wb'))

for _ in range(98):
    gp = SingleTaskGP(train_X=candidates, train_Y=scores)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    logEI = LogExpectedImprovement(model=gp, best_f=scores.max())
    gamma, _ = optimize_acqf(
        logEI,
        bounds=bounds,
        q=1,
        num_restarts=30, 
        raw_samples=1024
    )
    candidates = torch.cat([candidates, gamma], dim=0).to(torch.double)
    vb = SMI_VB(torch.hstack([gamma,gamma]).to(torch.float32),dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior)
    vb.train(iterations_training)
    scores = torch.cat([scores,score(vb).reshape((1,1))], dim=0).to(torch.double)
    pickle.dump({"candidates": candidates,'scores':scores},open('results/fits.p','wb'))
    pickle.dump(vb,open('results_application/smi_'+str(num_model)+'.p','wb'))
    num_model += 1

