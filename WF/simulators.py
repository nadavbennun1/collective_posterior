import numpy as np
import torch
# from inference_utils import get_prior
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior
)

def check_parameters(sim, parameters):
    if sim == 'WF':
        return len(parameters) == 3
    else:
        raise ValueError('Wrong number of parameters for WF simulator')


# constants
N = int(1e7)
generation = np.array([8, 21, 29, 37, 50, 58, 66, 79, 87, 95, 108, 116]) # from Chuong et al 2024

def WF(parameters, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for x generations
    Returns proportion of the population with a CNV for specific generations
    
    Parameters
    -------------------
    N : int = 10M
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations to output
    seed : int
    
    s_cnv : float
        fitness benefit of CNVs  
    m_cnv : float 
        probability mutation to CNV 
    p_0: float
        fraction of population with GAP1 CNV before beginning of experiment
    """
    assert check_parameters('WF', parameters)
    # CNV parameters
    s_cnv, m_cnv, p_0 = 10**parameters

    # SNV parameters as constants
    s_snv = 1e-3
    m_snv = 1e-5
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    
    
    # Order is: wt, cnv+, cnv-, snv
    
    w = np.array([1, 1 + s_cnv, 1 + s_cnv, 1 + s_snv], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m_cnv - m_snv, 0, 0, 0],
                [m_cnv, 1, 0, 0],
                [0, 0, 1, 0],
                [m_snv, 0, 0, 1]], dtype='float64')
    assert np.allclose(M.sum(axis=0), 1)
    
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(4)
    n[2] = N*p_0 # cnv-
    n[0] = N*(1-p_0) # wt
    
    # follow proportion of the population with CNV
    # here rows will be generation, columns (there is only one) is replicate population
    p_cnv = []
    
    # run simulation to generation 116
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_cnv.append(p[1])  # frequency of reported CNVs
        p = E @ p.reshape((4, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    
    return torch.tensor(np.transpose(p_cnv)[generation.astype(int)])


def WF_wrapper(reps, parameters, seed=None):
    evo_reps = torch.empty(reps, len(generation))
    for i in range(reps):
        out=WF(parameters, seed=seed)
        evo_reps[i,:] = out
    return evo_reps
