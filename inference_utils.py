import torch
from sbi.utils import BoxUniform

# Define the prior
def get_prior(sim):
    if sim == 'WF':
        prior = BoxUniform(low=torch.tensor([-2, -7, -8]), high=torch.tensor([0, -2, -2]))
    else:
        raise ValueError('Unknown simulator')
    return prior