# import torch
# from inference_utils import get_prior

def simulate(sim, parameters):
    assert check_parameters(sim, parameters)
    if sim == 'WF':
        return WF(parameters)
    else:
        raise ValueError('Unknown simulator')

def check_parameters(sim, parameters):
    if sim == 'WF':
        return len(parameters) == 3
    else:
        raise ValueError('Wrong number of parameters for WF simulator')
    
# def WF(parameters):
    # simulate the WF model as in Chuong et al. 2024
    # parameters: [s, m, p]
    # s: selection coefficient
    # m: mutation rate
    # p: initial unreported frequency of the beneficial allele
    # s, m, p = parameters
    
    # return torch.tensor([s, m, p])