# inference with NPE
from simulators import WF_wrapper, GLU_wrapper, SLCP_wrapper
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
parser.add_argument('-cp', "--cp", action='store_true') # whether to use collective posterior
args = parser.parse_args()


model_dict = {'GLU': GLU_wrapper, 'WF': WF_wrapper, 'SLCP': SLCP_wrapper}

# Define the prior and simulator
sim = str(args.model)
simulator = model_dict[sim]
thetas_dir = args.test_thetas
posterior_dir = args.posterior
samples = int(args.samples)
c = args.cp

# Load the posterior with pickle
posterior = pickle.load(open(posterior_dir, 'rb'))

# Load the test thetas
thetas = torch.tensor(pd.read_csv(thetas_dir, index_col=0).values.astype('float'), dtype=torch.float32)

def evaluate_cp(posterior, thetas, n_samples):
    accus = torch.empty(thetas.shape)
    covs = torch.empty(len(thetas),1)
    for i in range(len(thetas)):
        X = simulator(10, thetas[i])
        cp = CollectivePosterior(prior=get_prior(sim), amortized_posterior=posterior, log_C=1, Xs=X, epsilon=-10000)
        samples = cp.sample(n_samples)
        params = torch.tensor(thetas[i,:], dtype=torch.float32)
        accus[i] = samples.mean(0)-params
        covs[i] = (cp.log_prob(samples) > cp.log_prob(params)).sum()/len(thetas)
        if i%10 == 9:
          print(f'{round(100*(i+1)/len(thetas),2)}%')
    return accus, covs


def evaluate_iid(posterior, thetas, n_samples):
    accus = torch.empty(thetas.shape)
    covs = torch.empty(len(thetas),1)
    for i in range(len(thetas)):
        X = simulator(10, thetas[i])
        samples = posterior.set_default_x(X).sample((n_samples,))
        params = torch.tensor(thetas[i,:], dtype=torch.float32)
        accus[i] = samples.mean(0)-params
        covs[i] = (posterior.log_prob(samples) > posterior.log_prob(params)).sum()/len(thetas)
        if i%10 == 1:
          print(f'{round(100*i/len(thetas),2)}%')
    return accus, covs

eval_func = evaluate_cp if c else evaluate_iid
add_iid = '' if c else '_iid'

accus, covs = eval_func(posterior, thetas, n_samples=samples)
accus = accus.detach().numpy()
covs = covs.detach().numpy()
pd.DataFrame(accus).to_csv(f'{sim}/tests/accus_{sim}{add_iid}.csv')
pd.DataFrame(covs).to_csv(f'{sim}/tests/covs_{sim}{add_iid}.csv')