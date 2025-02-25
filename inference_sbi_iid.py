# inference with NPE
from simulators import WF_wrapper, GLU_wrapper, SLCP_wrapper
from sbi.utils import BoxUniform
import torch
import pickle
from time import time
import argparse
import sbibm
from sbi.inference import NPE, simulate_for_sbi

from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

# time
start = time()

# Define the prior and simulator
#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-e', "--epochs")
parser.add_argument('-n', "--num_sim")
args = parser.parse_args()

# Define the prior
def get_prior(sim):
    if sim == 'WF':
        prior = BoxUniform(low=torch.tensor([-2, -7, -8]), high=torch.tensor([0, -2, -2]))
    elif sim == 'GLU':
        prior = sbibm.get_task('gaussian_linear_uniform').get_prior_dist()
    elif sim == 'SLCP':
        prior = sbibm.get_task('slcp').get_prior_dist()
    else:
        raise ValueError('Unknown simulator')

    return prior


sim = str(args.model)
stop_after_epochs = int(args.epochs)
num_sim = int(args.num_sim)

prior = get_prior(sim)
model_dict = {'GLU': GLU_wrapper, 'WF': WF_wrapper, 'SLCP': SLCP_wrapper}
simulator = model_dict[sim]


max_num_trials = 10
  

# construct training data set: we want to cover the full range of possible number of
# trials
num_training_samples = num_sim
theta = prior.sample((num_training_samples,))

# there are certainly smarter ways to construct the training data set, but we go with a
# for loop here for illustration purposes.
x_dim_dict = {'GLU': 10, 'WF': 12, 'SLCP': 8} 
x_dim = x_dim_dict[sim]

x = torch.ones(num_training_samples * max_num_trials, max_num_trials, x_dim) * float(
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
    input_dim=x_dim,
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

inference = NPE(prior, density_estimator=density_estimator)
# NOTE: we don't exclude invalid x because we used NaNs for the missing trials.
inference.append_simulations(
    theta,
    x,
    exclude_invalid_x=False,
).train(stop_after_epochs=stop_after_epochs)
posterior = inference.build_posterior()

# Save the posterior with pickle
with open(f'{sim}/posteriors/posterior_iid_{sim}_{num_sim}_{stop_after_epochs}.pkl', 'wb') as f:
    pickle.dump(posterior, f)

# time
end = time()
print(f'Inference took {end - start} seconds')
