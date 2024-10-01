# inference with NPE
from simulators import WF
from inference_utils import get_prior
import torch
import pickle
import time
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


# time
start = time.time()

# Define the prior and simulator
sim = 'WF'
prior = get_prior(sim)
simulator = WF

# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(simulator, prior, prior_returns_numpy)

# inference
inference = NPE(prior)

theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=100000)
density_estimator = inference.append_simulations(theta, x).train(training_batch_size=50, stop_after_epochs=20,  num_workers=20)
posterior = inference.build_posterior(density_estimator)

# Save the posterior with pickle
with open(f'posterior_{sim}_20.pkl', 'wb') as f:
    pickle.dump(posterior, f)

# time
end = time.time()
print(f'Inference time: {end - start} seconds')
