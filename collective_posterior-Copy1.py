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
from time import time

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
        A = torch.prod(lens) # Prior volume3
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
        t = time()
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
        print(time()-t)
        return torch.cat(samples)

    def sample_one(self, jump=int(10**4), keep=True):
        sampled=False
        while not(sampled):
            samps = self.prior.sample((jump,))
            probs = torch.rand(samps.size()[0])
            lp = self.log_prob(samps)
            next_idx = lp > probs
            if next_idx.sum()>0:
                sampled=True
        return samps[next_idx][0]
    
    def sample_around(self, theta, jump=int(1e4), scale=0.05):
        dist =  torch.distributions.multivariate_normal.MultivariateNormal(theta, torch.diag(torch.tensor([scale]*self.theta_dim)))
        cands = dist.sample((jump,))
        probs = self.log_prob(cands)
        baseline = torch.rand((len(cands),))
        res = cands[probs>baseline]
        return res
    
    def sample(self, n_samples, jump=int(1e4), keep=True, scale=0.05):
        theta = self.sample_one(jump,keep=False)
        samples = torch.empty((n_samples,len(theta)))
        cur = 0
        while cur<n_samples:
            samps = self.sample_around(theta,jump,scale=scale)
            how_many = len(samps)
            if cur+how_many > n_samples:
                how_many = n_samples-cur
            if how_many>0:
                samples[cur:cur+how_many,:] = samps[:how_many,:]
                cur += how_many
        if keep:
            self.samples = samples
        return samples
    
    
    def sample_mcmc_lang(self,num_samples, step_size, burn_in=1000):
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
        # init_samples = self.prior.sample((burn_in,))
        # init_probs = self.log_prob(init_samples)
        # initial_theta = init_samples[init_probs.argmax()]
        initial_theta = self.sample_one()
        theta = initial_theta.clone()
        samples = []

        for i in range(num_samples + burn_in):
            # Propose a new sample from a Gaussian centered at the current theta
            proposal = theta + torch.randn_like(theta) * step_size

            # Compute log-probabilities for the current and proposed samples
            current_log_prob = self.log_prob(theta)
            proposal_log_prob = self.log_prob(proposal)

            # Compute the acceptance probability
            acceptance_prob = proposal_log_prob - current_log_prob
            
            print(torch.log(torch.rand(1)), acceptance_prob)
            # Accept or reject the proposal
            if torch.log(torch.rand(1)) < acceptance_prob.item():
                theta = proposal + torch.normal(0,0.01,size=(1,theta.size(0)))

            # Save the sample if we're past the burn-in period
            if i >= burn_in:
                samples.append(theta.clone())
            print(samples)

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
    
    def get_log_C(self, samples=int(1e5), n_reps=5):
        loglens = torch.log(torch.tensor([float(self.prior.base_dist.high[i])-float(self.prior.base_dist.low[i]) for i in range(len(self.prior.base_dist.high))])) # Prior dimensions
        logA = loglens.sum() # Prior volume
        log_dt = logA - torch.log(torch.tensor(samples)) # granularity
        eps = torch.tensor(self.epsilon, dtype=torch.float32)
        r = len(self.Xs)
        log_probs = torch.empty((samples, r))
        res = []
        for k in range(n_reps):
            g = self.prior.sample((samples,))
            for i in range(r):
                log_probs[:,i] = torch.max(self.amortized_posterior.set_default_x(self.Xs[i,:]).log_prob(g), eps)
            res.append(-1*torch.logsumexp(torch.sum(log_probs,-1)+ log_dt -(1-r)*logA ,0))
        self.log_C = torch.tensor(res).mean()
        return self.log_C
        
        
    def sample_around_(self, theta, jump=int(1e4), scale=0.5):
        dist =  torch.distributions.multivariate_normal.MultivariateNormal(theta, torch.diag(torch.tensor([scale]*self.theta_dim)))
        cands = dist.sample((jump,))
        probs = self.log_prob(cands)
        baseline = torch.rand((len(cands),))
        res = cands[probs>baseline]
        new_theta = cands[probs.argmax()]
        return res, new_theta
    
    def sample_(self, n_samples, jump=int(1e4), keep=True, scale=0.05):
        theta = self.sample_one(jump,keep=False)
        samples = torch.empty((n_samples,len(theta)))
        cur = 0
        while cur<n_samples:
            samps, theta = self.sample_around_(theta,jump,scale=scale)
            how_many = len(samps)
            if cur+how_many > n_samples:
                how_many = n_samples-cur
            if how_many>0:
                samples[cur:cur+how_many,:] = samps[:how_many,:]
                cur += how_many
        if keep:
            self.samples = samples
        return samples
    

        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import sbi.utils as utils
import torch
from sbi.inference import MCMCPosterior
from scipy.special import logsumexp
from seaborn import histplot

import sys  
sys.path.insert(1, '../')
from simulators import WF, WF_wrapper
generation = np.array(pd.read_csv('WF/empirical_data/Chuong_116_gens.txt').columns.astype('int'))


def simulator(parameters):
    X = WF_wrapper(parameters=parameters, seed=None, reps=10) \
    # + np.random.normal(0,0.01,(10,12))
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i,j] > 1:
                X[i,j] = 1
            if X[i,j] < 0:
                X[i,j] = 0
    return X

th = torch.tensor([-0.74,-4.84,-4.32], dtype=torch.float32)
X = simulator(th.numpy()) # LTRΔ MAP in paper

    
prior_min = np.log10(np.array([1e-2,1e-7,1e-8]))
prior_max = np.log10(np.array([1,1e-2,1e-2]))
prior = utils.BoxUniform(low=torch.tensor(prior_min), 
                         high=torch.tensor(prior_max))
posterior_chuong = pickle.load(open('WF/posteriors/posterior_WF_100000_100.pkl', 'rb'))
epsilon = -15000
Xs = torch.tensor(X, dtype=torch.float32)
op = CollectivePosterior(prior, posterior_chuong, Xs, 1, epsilon)

t = time()
print(op.get_log_C(), time()-t)
t = time()
op.amortized_posterior.set_default_x(Xs[0])

# print(op.amortized_posterior.posterior_estimator.net._mean)

plt.hist(op.sample(200, keep=False)[:,0])
print(time()-t)
t = time()
plt.hist(op.sample_(200)[:,0], alpha=0.2)
print(time()-t)
plt.show()
