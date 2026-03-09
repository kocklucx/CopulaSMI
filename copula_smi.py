import torch

class SMI_VB:
    def __init__(self,gamma,dim_psi,dim_eta,y,log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior):
        """
        Parameters
        ----------
        gamma: d-dimensional vector of influence parameters in [0,1]. 
        dim_psi: dimension of the copula parameter vector psi
        dim_eta : dimension of the stacked vector of marginal parameters eta
        y: (n,d)-dimensional array containing the observed data 
        log_pdf_copula: function evaluating the log copula-likelihood evaluated at auxillary parameters u of shape (n,d) and the copula parameter psi
        cdfs_marginals: function transforming the observed data y elementwise into the cdf values at a given marginal parameter eta
        log_pdf_marginals: function evaluating the log likelihood for the marginals; takes y and eta as input
        log_prior: evaluates the log-prior for (eta,psi). 

        Returns
        -------
        The function .train() optimizes the variational parameters. Then, .single_sample() can be used to generate a draw from q_\lambda(\psi,z,\eta,\tilde{\eta}).

        """
        self.gamma = gamma
        self.dim_psi = dim_psi
        self.dim_eta = dim_eta
        self.n = len(y)
        self.y = y
        #
        self.log_pdf_copula,self.cdfs_marginals,self.log_pdf_marginals,self.log_prior = log_pdf_copula,cdfs_marginals,log_pdf_marginals,log_prior
        # calculate ranks
        self.ranks = torch.tensor([[(y[:,d]<=y[l,d]).sum() for l in range(self.n)] for d in range(len(y[0]))]).T 
        # initialize variational parameters
        eta_hat, psi_hat = self.find_clever_start()
        self.zeta = torch.zeros(self.ranks.shape)
        self.omega_log = -1.6*torch.ones(self.ranks.shape)
        self.mu_psi = psi_hat
        self.mu_eta = eta_hat
        self.mu_eta_tilde = eta_hat.clone()
        self.t_psi_lt = torch.zeros(int(self.dim_psi*(self.dim_psi-1)/2))
        self.t_psi_diag = 3*torch.ones(self.dim_psi)
        self.t_eta_lt = torch.zeros(int(self.dim_eta*(self.dim_eta-1)/2))
        self.t_eta_diag = 3*torch.ones(self.dim_eta)
        self.t_eta_tilde_lt = torch.zeros(int(self.dim_eta*(self.dim_eta-1)/2))
        self.t_eta_tilde_diag = 3*torch.ones(self.dim_eta)
        self.t_psi_eta = torch.zeros((self.dim_psi,self.dim_eta))
        self.t_psi_eta_tilde = torch.zeros((self.dim_psi,self.dim_eta))
        #
        self.ix_psi = torch.tril_indices(self.dim_psi,self.dim_psi,offset=-1)
        self.ix_eta = torch.tril_indices(self.dim_eta,self.dim_eta,offset=-1)
        
    def find_clever_start(self):
        # finds starting values for the SGA algorithm
        eta = torch.zeros((self.dim_eta,1))
        psi = torch.zeros((self.dim_psi,1))
        eta.requires_grad = True
        psi.requires_grad = True
        optimizer = torch.optim.Adam([eta,psi],maximize=True,lr=0.01)
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = self.log_pdf_marginals(self.y,eta) + self.log_pdf_copula(self.ranks/(self.n+1),psi)
            loss.backward()
            optimizer.step()
        eta.requires_grad = False
        psi.requires_grad = False
        return eta, psi

    def single_sample(self):
        # draws a single sample from p_{\text{SMI},\gamma}(\psi,u,\eta,\widetilde{\eta}\mid \mathcal{D}), respecting the stop gradients
        t_psi = torch.diag(self.t_psi_diag.exp())
        t_psi[self.ix_psi[0],self.ix_psi[1]] = self.t_psi_lt
        t_eta = torch.diag(self.t_eta_diag.exp())
        t_eta[self.ix_eta[0],self.ix_eta[1]] = self.t_eta_lt
        t_eta_tilde = torch.diag(self.t_eta_tilde_diag.exp())
        t_eta_tilde[self.ix_eta[0],self.ix_eta[1]] = self.t_eta_tilde_lt
        standard_gaussian = torch.distributions.Normal(torch.tensor([0.]),torch.tensor([1.]))
        z = self.zeta+self.omega_log.exp()*standard_gaussian.rsample(self.ranks.shape).squeeze()
        eps_psi = torch.randn((self.dim_psi,1))
        psi = self.mu_psi + torch.linalg.solve_triangular(t_psi.T,eps_psi,upper=True)
        eps_eta = torch.randn((self.dim_eta,1))
        eta = self.mu_eta-torch.matmul(torch.linalg.solve_triangular(t_eta.T,self.t_psi_eta.T,upper=True),psi.clone().detach()-self.mu_psi.clone().detach())+ torch.linalg.solve_triangular(t_eta.T,eps_eta,upper=True)
        eps_eta_tilde = torch.randn((self.dim_eta,1))
        eta_tilde = self.mu_eta_tilde-torch.matmul(torch.linalg.solve_triangular(t_eta_tilde.T,self.t_psi_eta_tilde.T,upper=True),psi-self.mu_psi)+ torch.linalg.solve_triangular(t_eta_tilde.T,eps_eta_tilde,upper=True)
        return z, psi, eta, eta_tilde

    def log_q(self,psi,z,eta,eta_tilde):
        t_psi = torch.diag(self.t_psi_diag.exp())
        t_psi[self.ix_psi[0],self.ix_psi[1]] = self.t_psi_lt
        t_eta = torch.diag(self.t_eta_diag.exp())
        t_eta[self.ix_eta[0],self.ix_eta[1]] = self.t_eta_lt
        t_eta_tilde = torch.diag(self.t_eta_tilde_diag.exp())
        t_eta_tilde[self.ix_eta[0],self.ix_eta[1]] = self.t_eta_tilde_lt
        q_psi =  -self.dim_psi/2*torch.tensor(2*torch.pi).log()+self.t_psi_diag.sum()-0.5*torch.matmul(t_psi.T,psi-self.mu_psi).pow(2).sum()
        q_eta_mean = self.mu_eta-torch.linalg.solve_triangular(t_eta.T,self.t_psi_eta.T@(psi.clone().detach()-self.mu_psi.clone().detach()),upper=True)
        q_eta =  -self.dim_eta/2*torch.tensor(2*torch.pi).log()+self.t_eta_diag.sum()-0.5*torch.matmul(t_eta.T,eta-q_eta_mean).pow(2).sum()
        q_eta_tilde_mean = self.mu_eta_tilde-torch.linalg.solve_triangular(t_eta_tilde.T,self.t_psi_eta_tilde.T@(psi-self.mu_psi),upper=True)
        q_eta_tilde =  -self.dim_eta/2*torch.tensor(2*torch.pi).log()+self.t_eta_tilde_diag.sum()-0.5*torch.matmul(t_eta_tilde.T,eta_tilde-q_eta_tilde_mean).pow(2).sum()
        q_z = torch.distributions.Normal(self.zeta,self.omega_log.exp()).log_prob(z).sum()#
        return q_psi + q_z + q_eta + q_eta_tilde
    
    def train(self,max_epochs=10000,window_size=500):
        params = [self.zeta,self.omega_log,self.mu_psi,self.mu_eta,self.mu_eta_tilde,self.t_psi_lt,self.t_psi_diag,self.t_eta_lt,self.t_eta_diag,self.t_eta_tilde_lt,self.t_eta_tilde_diag,self.t_psi_eta,self.t_psi_eta_tilde]
        for parameter in params:
            parameter.requires_grad = True   
        optimizer = torch.optim.Adam(params,maximize=True,lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor=0.5,patience=5,min_lr=1e-4)
        elbo_track = 0
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            # ELBO
            z, psi, eta, eta_tilde = self.single_sample()
            cdf_tilde = self.cdfs_marginals(self.y,eta_tilde)
            a_tilde = self.gamma*cdf_tilde+(1-self.gamma)*(self.ranks-1)/(self.n+1)
            b_tilde = self.gamma*cdf_tilde+(1-self.gamma)*(self.ranks)/(self.n+1)
            u_tilde = (b_tilde-a_tilde)*torch.distributions.normal.Normal(0,1).cdf(z)+a_tilde
            elbo = self.log_pdf_copula(u_tilde,psi)+self.log_pdf_marginals(self.y,eta_tilde)+self.log_pdf_copula(self.cdfs_marginals(self.y,eta),psi.clone().detach())+self.log_pdf_marginals(self.y,eta)
            elbo += self.log_prior(psi.clone().detach(),eta)+self.log_prior(psi,eta_tilde)
            elbo -= self.log_q(psi,z,eta,eta_tilde)
            elbo += torch.distributions.Normal(torch.tensor([0.]),torch.tensor([1.])).log_prob(z).sum()
            # update
            elbo.backward()
            optimizer.step()
            with torch.no_grad():
                elbo_track += 1/window_size*elbo.item()
            # print 
            if epoch%window_size==0 and epoch>0:
                scheduler.step(elbo_track)
#                print(int(100*epoch/max_epochs),'%',elbo_track)
                elbo_track = 0
        for parameter in params:
            parameter.requires_grad = False
