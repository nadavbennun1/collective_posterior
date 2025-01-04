import torch
from sbi.utils import BoxUniform
import sbibm

# Define the prior
def get_prior(sim):
    if sim == 'WF':
        prior = BoxUniform(low=torch.tensor([-2, -7, -8]), high=torch.tensor([0, -2, -2]))
    elif sim == 'GLU':
        theta_dim = 10
        prior_min = torch.tensor([-1]*theta_dim, dtype=torch.float32)
        prior_max = torch.tensor([1]*theta_dim, dtype=torch.float32)
        prior = BoxUniform(low=prior_min, high=prior_max)
    elif sim == 'SLCP':
        theta_dim = 5
        prior_min = torch.tensor([-3]*theta_dim, dtype=torch.float32)
        prior_max = torch.tensor([3]*theta_dim, dtype=torch.float32)
        prior = BoxUniform(low=prior_min, high=prior_max)
    else:
        raise ValueError('Unknown simulator')

    return prior