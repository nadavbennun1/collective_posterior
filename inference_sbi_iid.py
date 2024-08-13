# inference with NPE
from simulators import WF, WF_wrapper
from sbi.inference import SNPE
from inference_utils import get_prior
import torch
import pickle
from time import time

# time
start = time()

# Define the prior and simulator
sim = 'WF'
prior = get_prior(sim)
simulator = WF_wrapper


# we need to fix the maximum number of trials.
max_num_trials = 10

# construct training data set: we want to cover the full range of possible number of
# trials
num_training_samples = 10000
theta = prior.sample((num_training_samples,))

# there are certainly smarter ways to construct the training data set, but we go with a
# for loop here for illustration purposes.
theta_dim = 12
x = torch.ones(num_training_samples * max_num_trials, max_num_trials, theta_dim) * float(
    "nan"
)
for i in range(num_training_samples):
    xi = simulator(reps=max_num_trials, parameters=theta[i])
    for j in range(max_num_trials):
        x[i * max_num_trials + j, : j + 1, :] = xi[: j + 1, :]

theta = theta.repeat_interleave(max_num_trials, dim=0)

# inference
from sbi.neural_nets.embedding_nets import (
    FCEmbedding,
    PermutationInvariantEmbedding,
)
from sbi.utils import posterior_nn

# embedding
latent_dim = 10
single_trial_net = FCEmbedding(
    input_dim=theta_dim,
    num_hiddens=40,
    num_layers=2,
    output_dim=latent_dim,
)
embedding_net = PermutationInvariantEmbedding(
    single_trial_net,
    trial_net_output_dim=latent_dim,
    # NOTE: post-embedding is not needed really.
    num_layers=1,
    num_hiddens=10,
    output_dim=10,
)

# we choose a simple MDN as the density estimator.
# NOTE: we turn off z-scoring of the data, as we used NaNs for the missing trials.
density_estimator = posterior_nn("maf", embedding_net=embedding_net, z_score_x="none")

inference = SNPE(prior, density_estimator=density_estimator)
# NOTE: we don't exclude invalid x because we used NaNs for the missing trials.
inference.append_simulations(
    theta,
    x,
    exclude_invalid_x=False,
).train(training_batch_size=1000)
posterior = inference.build_posterior()

# Save the posterior with pickle
with open(f'posterior_iid_{sim}.pkl', 'wb') as f:
    pickle.dump(posterior, f)

# time
end = time()
print(f'Inference took {end - start} seconds')