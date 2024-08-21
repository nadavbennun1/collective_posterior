# inference with NPE
from simulators import WF
from sbi.inference.base import infer
from inference_utils import get_prior
import torch
import pickle
import time
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


# time
start = time.time()

# Define the prior and simulator
sim = 'WF'
prior = get_prior(sim)
simulator = WF

# inference
simulator, prior = prepare_for_sbi(simulator, prior)
inference = SNPE(prior)

theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=100000)
density_estimator = inference.append_simulations(theta, x).train(training_batch_size=50, stop_after_epochs=100)
posterior = inference.build_posterior(density_estimator)

# Save the posterior with pickle
with open(f'posterior_{sim}.pkl', 'wb') as f:
    pickle.dump(posterior, f)

# time
end = time.time()
print(f'Inference time: {end - start} seconds')
