import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import sbi.utils as utils
import torch
from sbi.inference import ImportanceSamplingPosterior
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import pearsonr as pearson
from seaborn import pairplot
# pyro sampler - importance sampling
from pyro.infer import Importance

class CollectivePosterior:
       
    def __init__(self, prior, amortized_posterior, Xs, n_eval, log_C, epsilon):
        self.prior = prior # inference prior
        self.amortized_posterior = amortized_posterior
        self.Xs = Xs # Observations
        self.n_eval = n_eval # Granularity
        self.log_C = log_C # Normalizing constant, can be re-calculated using get_log_C
        self.epsilon = epsilon # Sensitivity, minimal reported value for a single observation by the amortized posterior
        self.map = None
        self.samples = []
    
    # Evaluate the collective posterior's log probability for a specific parameter set
    def log_prob(self, theta):
        theta = torch.tensor(theta, dtype=torch.float32)
        # allow one-dim tensor for theta
        if len(theta.size()) == 1:
            theta_size = 1
        else:
            theta_size = theta.size()[0]
        
        # get number of reps
        r = len(self.Xs)
        posterior = self.amortized_posterior
            
        # epsilon must be a tensor
        eps = torch.tensor(self.epsilon)
        if type(theta) != type(torch.tensor(4.2)):
            theta = torch.tensor(theta, dtype=torch.float32)
        if len(theta.size()) > 1:
            t = theta.size()[0]
            eps = torch.tensor([eps for i in range(t)])

        # Get log_prob value
        log_probs = torch.empty((theta_size,r))
        for i in range(r):
            log_probs[:,i] = torch.max(eps,posterior.set_default_x(self.Xs[i,:]).log_prob(theta))
        lens = self.prior.base_dist.high - self.prior.base_dist.low # Prior dimensions
        A = torch.prod(lens) # Prior volume
        return log_probs.sum(axis=1) + self.log_C - (1-r)*torch.log(A) # log rules
    
    
    def individual_log_probs(self, theta):
        # get number of reps
        r = len(self.Xs)

        # switch to MCMC-posterior for faster calculations
        potential_fn = self.amortized_posterior.potential_fn
        posterior_imp = ImportanceSamplingPosterior(potential_fn, proposal = self.prior)
        
        # epsilon must be a tensor
        eps = torch.tensor(self.epsilon)
        if type(theta) != type(torch.tensor(4.2)):
            theta = torch.tensor(theta, dtype=torch.float32)
        if len(theta.size()) > 1:
            t = theta.size()[0]
            eps = torch.tensor([eps for i in range(t)])

        # Get log_prob value
        log_probs = [float(posterior_imp.set_default_x(self.Xs[i,:]).log_prob(theta)) for i in range(r)]
        lens = np.array([float(self.prior.base_dist.high[i])-float(self.prior.base_dist.low[i]) for i in range(len(self.prior.base_dist.high))]) # Prior dimensions
        A = np.prod(lens) # Prior volume
        return np.array(log_probs) #+ float(self.log_C + np.log((1/A)**(1-r))) / r
    
    # Sample from the collective posterior using rejection sampling
#     def rejection_sample(self, n_samples, jump = int(10**5), keep=True):
#         samples = torch.empty((n_samples,3))
#         cur = 0
#         while cur<n_samples:
#             samps = self.prior.sample((jump,))
#             probs = torch.rand(samps.size()[0])
#             lp = self.log_prob(samps)
#             next_idx = lp > probs
#             how_many = next_idx.sum()
#             print(how_many)
#             if cur+how_many > n_samples:
#                 how_many = n_samples-cur-1
#             if how_many>0:
#                 samples[cur:cur+how_many,:] = samps[next_idx]
#                 cur += how_many
#         if keep:
#             self.samples = samples
#         return samples
    
    def sample_one(self, jump=int(10**5), keep=True):
        sampled=False
        while not(sampled):
            samps = self.prior.sample((jump,))
            probs = torch.rand(samps.size()[0])
            lp = self.log_prob(samps)
            next_idx = lp > probs
            if next_idx.sum()>0:
                sampled=True
        return samps[next_idx][0]
    
    def sample_around(self, theta, jump):
        dist =  torch.distributions.multivariate_normal.MultivariateNormal(theta, torch.diag(torch.tensor([0.05,0.05,0.05])))
        cands = dist.sample((jump,))
        probs = self.log_prob(cands)
        baseline = torch.rand((len(cands),))
        res = cands[probs>baseline]
        return res
    
    def sample(self, n_samples, jump=int(1e5), keep=True):
        theta = self.sample_one(jump,keep=False)
        samples = torch.empty((n_samples,len(theta)))
        cur = 0
        while cur<n_samples:
            samps = self.sample_around(theta,jump)
            how_many = len(samps)
            if cur+how_many > n_samples:
                how_many = n_samples-cur
            if how_many>0:
                samples[cur:cur+how_many,:] = samps[:how_many,:]
                cur += how_many
        if keep:
            self.samples = samples
        return samples
    
    # sample with MCMC with 1 rejection sample as initiator
    # def sample_mcmc(self, n_samples, keep=True):
    #     init = self.rejection_sample(1,keep=False)[0]
    #     print(init)
    #     samples = torch.empty((n_samples,3))
    #     samples[0,:] = init
    #     cand = init
    #     for i in range(1,n_samples):
    #         samp = self.sample_around(cand,1)[0]
    #         log_prob_samp = self.log_prob(samp)
    #         log_prob_cand = self.log_prob(cand)
    #         if log_prob_samp > log_prob_cand:
    #             samples[i,:] = samp
    #             cand = samp
    #         else:
    #             if torch.rand(1) < torch.exp(log_prob_samp - log_prob_cand):
    #                 samples[i,:] = samp
    #                 cand = samp
    #             else:
    #                 samples[i,:] = cand
    #     if keep:
    #         self.samples = samples
    #     return samples
    
    # get Maximum A-Posteriori (MAP) - distribution's mode
    def get_map(self, keep=True):
        func = lambda theta: np.array(-1*self.log_prob(theta)) # max(log_prob) = min(-log_prob) 
        if len(self.samples) > 0:
            x0 = self.samples[0,:]
        else:
            x0 = self.sample(n_samples=1, keep=False)[0,:] # get a nice random guess
        x0 = list(x0)
        collective_map = minimize(func, x0, method='Nelder-Mead').x # Scipy magic
        if keep:
            self.map = collective_map
        return collective_map
        
    
    # Plot marginal and pairwise-marginal distributions (based on existing samples) - Only relevant to Chuong et. al, 2024
    def plot_pairwise(self, color): 
        # credit to https://stackoverflow.com/questions/50832204/show-correlation-values-in-pairplot-using-seaborn-in-python
        def corrfunc(x, y, ax=None, hue=None, quantiles=[0.025, 0.975], **kws):
            """Plot the correlation coefficient in the top left hand corner of a plot."""
            #x = x[x>x.quantile(quantiles[0])][x<x.quantile(quantiles[1])]
            #y = y[y>y.quantile(quantiles[0])][y<y.quantile(quantiles[1])]
            r, _ = pearson(x, y)
            ax = ax or plt.gca()
            ax.set_title(f'ρ = {round(r,2)}', fontsize = 12)
            return
    
        # Validations
        assert len(self.samples) > 0
        posterior_samples = 10**self.samples
        if len(posterior_samples) < 30:
            print('You are using less than 30 samples. It is a bit risky to draw conclusions from that...')
        
        num_bins = 20            
        columns = ['$s_C$','$δ_C$','$φ$']
        g = pairplot(pd.DataFrame(posterior_samples, columns = columns), 
                     corner = True, plot_kws = {'color': color, 'levels':4}, diag_kws = {'color': color, 'bins':num_bins}, kind = 'kde', diag_kind='hist')
        g.fig.set_size_inches(9,6)
                
        # Titles, colors, HDIs, etc.
        labels = [f'{columns[0]} MAP = {round(float(10**self.map[0]),2)}', f'{columns[1]} MAP = $10^{ {round(float(self.map[1]),2)} }$', f'{columns[2]} MAP = $10^{ {round(float(self.map[2]),2)} }$']
        map_label='\n'.join(labels)
        
        for j in range(len(self.prior.base_dist.low)):
            g.axes[j,j].axvline(posterior_samples[:,j].quantile(0.975), color='blue')
            g.axes[j,j].axvline(posterior_samples[:,j].quantile(0.025), color='blue')
            g.axes[j,j].axvline(10**self.map[j], color='red', linewidth=3)
            if j==2:
                g.axes[j,j].axvline(posterior_samples[:,j].quantile(0.025), label='95% HDI', color='blue')
                g.axes[j,j].axvline(10**self.map[j], color='red', label=map_label, linewidth=3)

                
        #     if j>0:
        #         g.axes[2,j].set_xscale('log')
        #         g.axes[j,0].set_yscale('log')
        
                       

        
        g.fig.legend(fontsize=12, loc=(0.6, 0.7))
        g.map_lower(corrfunc)
        g.figure.tight_layout(pad=1)
               
        return g
    
    def get_log_C(self, samples=int(1e5)):
        lens = torch.tensor([float(self.prior.base_dist.high[i])-float(self.prior.base_dist.low[i]) for i in range(len(self.prior.base_dist.high))]) # Prior dimensions
        A = torch.prod(lens) # Prior volume
        dt = A / (self.n_eval**3) # granularity
        eps = torch.tensor(self.epsilon, dtype=torch.float32)
        r = len(self.Xs)
        log_probs = torch.empty((samples, r))
        res = []
        for k in range(10):
            g = self.prior.sample((samples,))
            for i in range(r):
                log_probs[:,i] = torch.max(self.amortized_posterior.set_default_x(self.Xs[i,:]).log_prob(g), eps)
            res.append(-1*torch.logsumexp(torch.sum(log_probs,-1)+ torch.log(dt) + torch.log((1/A)**(1-r)),0))
        self.log_C = torch.tensor(res).min()
        return self.log_C
        
        
        
#         def get_grid(posterior, x, n, epsilon):
#             # Explored space
#             s = np.linspace(self.prior.base_dist.low[0], self.prior.base_dist.high[0], n)
#             m = np.linspace(self.prior.base_dist.low[1], self.prior.base_dist.high[1], n)
#             p = np.linspace(self.prior.base_dist.low[2], self.prior.base_dist.high[2], n)

#             # Change to IMP posterior to sample more efficiently
#             potential_fn = self.amortized_posterior.potential_fn
#             posterior_imp = ImportanceSamplingPosterior(potential_fn, proposal = self.prior)

#             # Create empty grid
#             grd = torch.tensor([[[[s_,m_,p_,0] for s_ in s] for m_ in m] for p_ in p], dtype=torch.float32).reshape(n**3,4)
#             grd[:,3] = posterior_imp.log_prob(x=x,theta=grd[:,0:3])
#             grd[:,3] = grd[:,3].apply_(lambda x: max(epsilon,x))
#             return grd
        
#         x = self.Xs
#         r = len(x)
#         # get probs for first
#         rim = get_grid(self.amortized_posterior, x.iloc[0,:], self.n_eval,self.epsilon)
#         prod_df = pd.DataFrame(columns = ['log_s','log_m','log_p'] + list(x.index)+['sum_logs'], index=[i for i in range(len(rim))])
#         prod_df.iloc[:,0:3] = rim[:,0:3]
#         prod_df.iloc[:,3] = rim[:,3]
#         # insert other replicates to df
#         for i in range(1,len(x)):
#             x_0 = x.iloc[i,:]
#             prod_df.iloc[:,3+i] = get_grid(self.amortized_posterior,x_0,self.n_eval,self.epsilon)[:,3]
#         # Column of sum of log-posteriors = log(product of posteriors)
#         prod_df.loc[:,'sum_logs'] = prod_df.loc[:,list(x.index)].sum(axis=1)

#         # Calculate constants for the integral
#         lens = np.array([float(self.prior.base_dist.high[i])-float(self.prior.base_dist.low[i]) for i in range(len(self.prior.base_dist.high))]) # Prior dimensions
#         A = np.prod(lens) # Prior volume
#         dt = A / (self.n_eval**3) # granularity

#         # Obtain C using integral (approximated by Riemann sum)

#         # log(integrand) = log(product of posteriors * prior^(1-n) * dt)
#         prod_df['adj_sum_logs'] = prod_df['sum_logs'] + np.log(dt) + np.log((1/A)**(1-r))
#         # Riemann sum (minus -> inverse)
#         log_C = -1*logsumexp(prod_df['adj_sum_logs'].astype('float'))
#         self.log_C = log_C
#         return log_C