# inference with NPE
from simulators import WF
from sbi.inference.base import infer
from inference_utils import get_prior
import torch
import pickle
import time

# time
start = time.time()

# Define the prior and simulator
sim = 'WF'
prior = get_prior(sim)
simulator = WF

# inference
posterior = infer(simulator, prior, method='SNPE', num_simulations=10000)

# Save the posterior with pickle
with open(f'posterior_{sim}.pkl', 'wb') as f:
    pickle.dump(posterior, f)

# time
end = time.time()
print(f'Inference time: {end - start} seconds')
