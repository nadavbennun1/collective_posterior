# inference with NPE
from simulators import WF, GLU, SLCP, wrapper, wrapper_hierarchical
import torch
import pickle
# import time
import argparse
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import numpy as np
import pandas as pd
from collective_posterior import CollectivePosterior
from inference_utils import get_prior

#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model") # model name
parser.add_argument('-p', "--posterior") # posterior directory
parser.add_argument('-s', "--samples") # number of samples from the posterior
parser.add_argument('-t', "--test_thetas") # test thetas directory
parser.add_argument('-hi', "--hierarchical", action='store_true') # whether to test on hierarchical model
parser.add_argument('-cp', "--cp", action='store_true') # whether to use collective posterior
parser.add_argument('-e', "--ensemble", action='store_true') # whether to test on an ensemble
args = parser.parse_args()


model_dict = {'GLU': GLU, 'WF': WF, 'SLCP': SLCP}

# Define the prior and simulator
sim = str(args.model)
simulator = model_dict[sim]
thetas_dir = args.test_thetas
posterior_dir = args.posterior
samples = int(args.samples)
c = args.cp
h = args.hierarchical
e = args.ensemble

# Load the posterior with pickle
posterior = pickle.load(open(posterior_dir, 'rb'))

# Load the test thetas
thetas = torch.tensor(pd.read_csv(thetas_dir, index_col=0).values.astype('float'), dtype=torch.float32)


def coverage_old(posterior, samples, conf_levels, theta):
    covs = torch.empty(len(conf_levels), len(theta))
    sorted_probs, ind = torch.sort(posterior.log_prob(samples), descending=True)
    for j in range(len(conf_levels)):
        conf_level = conf_levels[j]   
        taken_samples = samples[:int(conf_level*len(samples))+1]
        # hdi = [torch.quantile(samples, (1-conf_level)/2, 0), torch.quantile(samples,(1+conf_level)/2, 0)]
        hdi = [taken_samples.min(0).values, taken_samples.max(0).values]
        covs[j,:] = ((theta > hdi[0])*(theta < hdi[1]))
    
    return covs

    

def evaluate_cp(posterior, thetas, n_samples):
    accus = torch.empty(thetas.shape)
    covs = torch.empty(len(thetas),1)
    covs_old = torch.empty(len(thetas[:,0]),4, len(thetas[0]))
    for i in range(len(thetas)):
        if h:
            X = wrapper_hierarchical(simulator, 10, thetas[i])
        else:
            X = wrapper(simulator, 10, thetas[i])
        cp = CollectivePosterior(prior=get_prior(sim), amortized_posterior=posterior, log_C=1, Xs=X, epsilon=-150)
        cp.get_log_C()
        samples = cp.sample(n_samples)
        params = torch.tensor(thetas[i,:], dtype=torch.float32)
        accus[i] = samples.mean(0)-params
        covs[i] = (cp.log_prob(samples) > cp.log_prob(params)).sum()/n_samples
        covs_old[i] = coverage_old(posterior, samples, conf_levels=[0.5,0.8,0.9,0.95], theta=thetas[i])
        print(i)
        if i%10 == 9:
          print(f'{round(100*(i+1)/len(thetas),2)}%')
    return accus, covs, covs_old

def evaluate_iid(posterior, thetas, n_samples):
    accus = torch.empty(thetas.shape)
    covs = torch.empty(len(thetas),1)
    covs_old = torch.empty(len(thetas[:,0]),4, len(thetas[0]))
    for i in range(len(thetas)):
        if h:
            X = wrapper_hierarchical(simulator, 10, thetas[i])
        else:
            X = wrapper(simulator, 10, thetas[i])
        samples = posterior.set_default_x(X).sample((n_samples,))
        params = torch.tensor(thetas[i,:], dtype=torch.float32)
        accus[i] = samples.mean(0)-params
        covs[i] = (posterior.log_prob(samples) > posterior.log_prob(params)).sum()/len(thetas)
        covs_old[i] = coverage_old(posterior, samples, conf_levels=[0.5,0.8,0.9,0.95], theta=thetas[i])
        if i%10 == 1:
          print(f'{round(100*i/len(thetas),2)}%')
    return accus, covs, covs_old

eval_func = evaluate_cp if c else evaluate_iid
add_iid = '' if c else '_iid'
add_h = '_h' if h else ''
add_e = '_e' if e else ''


accus, covs, covs_old = eval_func(posterior, thetas, n_samples=samples)
covs_old = covs_old.mean(0)
accus = accus.detach().numpy()
covs = covs.detach().numpy()
covs_old = covs_old.detach().numpy()
pd.DataFrame(accus).to_csv(f'{sim}/tests/accus_{sim}{add_iid}{add_h}{add_e}.csv')
pd.DataFrame(covs).to_csv(f'{sim}/tests/covs_{sim}{add_iid}{add_h}{add_e}.csv')
pd.DataFrame(covs_old, index=[0.5,0.8,0.9,0.95]).to_csv(f'{sim}/tests/covs_old_{sim}{add_iid}{add_h}{add_e}.csv')