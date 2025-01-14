import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import sbi.utils as utils
import torch
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import pearsonr as pearson
from seaborn import pairplot

class CollectivePosterior:
       
    def __init__(self, prior, amortized_posterior, Xs, log_C, epsilon):
        self.prior = prior # inference prior
        self.amortized_posterior = amortized_posterior
        self.Xs = Xs # Observations
        self.log_C = log_C # Normalizing constant, can be re-calculated using get_log_C
        self.epsilon = epsilon # Sensitivity, minimal reported value for a single observation by the amortized posterior
        self.map = None
        self.samples = []
        self.theta_dim = len(prior.base_dist.low)
        self.n_eval = int(1e5)
    
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

        logpr = self.prior.log_prob(theta)
        return log_probs.sum(axis=1) + self.log_C - (1-r)*logpr # log rules
    
    
    
    # Sample from the collective posterior using rejection sampling
    def rejection_sample(self, n_samples, jump = int(10**5), keep=True):
        samples = torch.empty((1,self.theta_dim))
        cur = 0
        while cur<n_samples+1:
            samps = self.prior.sample((jump,))
            probs = torch.rand(samps.size()[0])
            lp = self.log_prob(samps)
            next_idx = lp > probs
            samples_to_add = samps[next_idx]
            samples = torch.cat([samples, samples_to_add])
            cur += next_idx.sum()
            if keep:
                self.samples = samples[1:n_samples]
        return samples[1:n_samples]
    
    def rejection_sampler(self, num_samples, jump = int(1e5)):
        """
        Rejection sampler for sampling from a target distribution using log-probabilities.

        Parameters:
            log_prob (function): Function computing the log-probability of the target distribution.
            proposal_sampler (function): Function to sample from the proposal distribution.
            log_q (function): Function computing the log-probability of the proposal distribution.
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Samples from the target distribution.
        """
        samples = []
        while len(samples) < num_samples:
            # Sample from the proposal distribution
            proposal = self.prior.sample((jump,))

            # Compute acceptance probability in log-space
            log_target_prob = self.log_prob(proposal)
            log_prop_prob = self.prior.log_prob(proposal)
            log_acceptance_prob = log_target_prob - log_prop_prob

            # Accept or reject the proposal
            accs = proposal[torch.rand(jump).log() < log_acceptance_prob]
            if len(accs) > 0:
                samples.append(accs)

        return torch.cat(samples)

    def sample_one(self, jump=int(1e4), keep=True):
        sampled=False
        while not(sampled):
            samps = self.prior.sample((jump,))
            probs = torch.rand(samps.size()[0])
            lp = self.log_prob(samps)
            next_idx = lp > probs
            if next_idx.sum()>0:
                sampled=True
        return samps[next_idx][0]
    
    def sample_around(self, theta, jump=int(1e4)):
        dist =  torch.distributions.multivariate_normal.MultivariateNormal(theta, torch.diag(torch.tensor([0.05]*self.theta_dim)))
        cands = dist.sample((jump,))
        probs = self.log_prob(cands)
        baseline = torch.rand((len(cands),))
        res = cands[probs>baseline]
        new_theta = cands[probs.argmax()]
        return res, new_theta
    
    def sample(self, n_samples, jump=int(1e4), keep=True):
        theta = self.sample_one(jump,keep=False)
        samples = torch.empty((n_samples,len(theta)))
        cur = 0
        while cur<n_samples:
            samps, theta = self.sample_around(theta,jump)
            how_many = len(samps)
            if cur+how_many > n_samples:
                how_many = n_samples-cur
            if how_many>0:
                samples[cur:cur+how_many,:] = samps[:how_many,:]
                cur += how_many
        if keep:
            self.samples = samples
        return samples
    
    
    def sample_mcmc(self,num_samples, step_size, burn_in=1000):
        """
        MCMC sampler using the Metropolis-Hastings algorithm.

        Parameters:
            log_prob (function): Function computing the log-probability of theta.
            initial_theta (torch.Tensor): Initial position in parameter space.
            num_samples (int): Number of samples to draw.
            step_size (float): Standard deviation of the Gaussian proposal distribution.
            burn_in (int): Number of initial samples to discard.

        Returns:
            torch.Tensor: Samples from the target distribution.
        """
        # Initialize
        init_samples = self.prior.sample((burn_in,))
        init_probs = self.log_prob(init_samples)
        initial_theta = init_samples[init_probs.argmax()]
        theta = initial_theta.clone()
        samples = []

        for i in range(num_samples + burn_in):
            # Propose a new sample from a Gaussian centered at the current theta
            proposal = theta + torch.randn_like(theta) * step_size

            # Compute log-probabilities for the current and proposed samples
            current_log_prob = self.log_prob(theta)
            proposal_log_prob = self.log_prob(proposal)

            # Compute the acceptance probability
            acceptance_prob = torch.exp(proposal_log_prob - current_log_prob)

            # Accept or reject the proposal
            if torch.rand(1).item() < acceptance_prob.item():
                theta = proposal

            # Save the sample if we're past the burn-in period
            if i >= burn_in:
                samples.append(theta.clone())

        return torch.stack(samples)


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
    
    def get_log_C(self, n_reps = 10):
        # loglens = torch.log(torch.tensor([float(self.prior.base_dist.high[i])-float(self.prior.base_dist.low[i]) for i in range(len(self.prior.base_dist.high))])) # Prior dimensions
        # logA = loglens.sum() # Prior volume
        # log_dt = logA - torch.log(torch.tensor(samples)) # granularity
        eps = torch.tensor(self.epsilon, dtype=torch.float32)
        r = len(self.Xs)
        log_probs = torch.empty((self.n_eval, r))
        res = []
        logdt = torch.log(torch.tensor(self.n_eval, dtype=torch.float32))
        for k in range(n_reps):
            prior_samples = self.prior.sample((self.n_eval,))
            prior_logps = self.prior.log_prob(prior_samples)
            for i in range(r):
                log_probs[:,i] = torch.max(self.amortized_posterior.set_default_x(self.Xs[i,:]).log_prob(prior_samples), eps)
            res.append(-1*torch.logsumexp(torch.sum(log_probs,-1) -(1-r)*prior_logps -1*logdt ,0))
        self.log_C = torch.tensor(res).mean()
        return self.log_C
        
    # def get_log_C_adaptive(self, samples = 1000, n_reps = 10, n_rounds = 3, n_sample_each_round = 100):
    #     for _ in range(n_rounds):
    #         init_log_c = self.get_log_C(samples, n_reps)
    #         round_samps = self.sample(n_sample_each_round, keep=False)
    #         print(round_samps.max(0)[0], round_samps.min(0)[0])
    #         loglens = torch.log(round_samps.max(0)[0] - round_samps.min(0)[0]) # round's lengths
    #         logA = loglens.sum() # round's volume
    #         log_dt = logA - torch.log(torch.tensor(n_sample_each_round)) # round's granularity
    #         eps = torch.tensor(self.epsilon, dtype=torch.float32)
    #         r = len(self.Xs)
    #         log_probs = torch.empty((n_sample_each_round, r))
    #         res = []
    #         for k in range(n_reps):
    #             for i in range(r):
    #                 log_probs[:,i] = torch.max(self.amortized_posterior.set_default_x(self.Xs[i,:]).log_prob(round_samps), eps)
    #             res.append(-1*torch.logsumexp(torch.sum(log_probs,-1)+ log_dt -(1-r)*logA ,0))
    #         self.log_C = torch.tensor(res).mean()
    #         print(self.log_C)
    #     return self.log_C