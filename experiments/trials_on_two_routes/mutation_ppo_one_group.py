import matplotlib.pyplot as plt
import os
import pandas as pd
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
import torch
from torchrl.collectors import SyncDataCollector
from torch.distributions import Categorical
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm
import sys
import os
import json
from keychain import Keychain as kc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environment.environment import TrafficEnvironment
from services.plotter import Plotter
from utilities import get_params

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Devices
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("device is: ", device)
vmas_device = device  # The device where the simulator is run

# Sampling
frames_per_batch = 50  # Number of team frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations - the episodes the plotter plots
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 100  # Number of optimization steps per training iteration
minibatch_size = 2  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-5  # coefficient of the entropy term in the PPO loss

############ Environment creation

params = get_params(kc.PARAMS_PATH)

env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS], params[kc.PHASE])

env.start()
env.reset()

############ Human learning

num_episodes = 100

for episode in range(num_episodes):
    env.step()


############ Mutation

env.mutation()

############ Machine learning

env = PettingZooWrapper(
    env=env,
    use_mask=True,
    categorical_actions=True,
    done_on_any = False,
    device=device
)

print("action_spec:", env.full_action_spec)
print("reward_spec:", env.full_reward_spec)
print("done_spec:", env.full_done_spec)
print("observation_spec:", env.observation_spec)

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

check_env_specs(env)

reset_td = env.reset()


############ Policy network


share_parameters_policy = False  # Can change this based on the group

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[
            -1
        ],  # n_obs_per_agent
        n_agent_outputs= env.full_action_spec["action"].space.n,  # n_actions_per_agents
        n_agents=env.n_agents,  # Number of agents in the group
        centralised=False,  # the policies are decentralised (i.e., each agent will act from its local observation)
        share_params=share_parameters_policy,
        device=device,
        depth=3,
        num_cells=64,
        activation_class=torch.nn.Tanh,
    ),
)

    # Wrap the neural network in a :class:`~tensordict.nn.TensorDictModule`.
    # This is simply a module that will read the ``in_keys`` from a tensordict, feed them to the
    # neural networks, and write the
    # outputs in-place at the ``out_keys``.

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "logits")],
)  # We just name the input and output that the network will read and write to the input tensordict

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.unbatched_action_spec,
    in_keys=[("agents", "logits")],
    out_keys=[("agents", "action")],
    distribution_class=Categorical,
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob"),
)


############ Critic network
share_parameters_critic = False
mappo = False  # IPPO if False

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1, 
    n_agents=env.n_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=4,
    num_cells=64,
    activation_class=torch.nn.Tanh,
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
)




############ Collector

collector = SyncDataCollector(
    env,
    policy,
    device=device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    reset_at_each_iter=False,
    total_frames=total_frames,
) 


############ Replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this size
)



############ PPO loss function

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)
loss_module.set_keys( 
    reward=env.reward_key,  
    action=env.action_key, 
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
    advantage=("agents", "advantage")
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
) 

GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

############ Training loop

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []

for tensordict_data in collector: ##loops over frame_per_batch

    # Update done and terminated for both agents
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", "agents", "reward"))),  # Adjust index to start from 0
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", "agents", "reward"))),  # Adjust index to start from 0
    )

    # Compute GAE for both agents
    with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

    data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                #print("Inside inner loop", subdata, replay_buffers, "\n\n")
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_grad_norm
                )  # Optional

                optim.step()
                optim.zero_grad()

    # Update policy weights for both agents
    #for group, _agents in env.group_map.items():
    collector.update_policy_weights_()
   

    done = tensordict_data.get(("next", "agents", "done"))  # Get done status for the group

    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "reward"))[
            tensordict_data.get(("next", "agents", "done"))
        ]
        .mean()
        .item()
    )

    episode_reward_mean_list.append(episode_reward_mean)

    pbar.set_description(
        ", ".join(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    )
    pbar.update()


############ Save
# Mean episode reward
"""
episode_reward_mean_file = kc.RECORDS_FOLDER + '/episode_reward_mean.json'
with open(episode_reward_mean_file, 'w') as f:
    json.dump(episode_reward_mean_map, f)

# Total loss
loss_file = kc.RECORDS_FOLDER + '/loss_file.json'
# loss is a tensor so I transform it to a list
loss = {group: [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
        for group, tensors in loss.items()}
with open(loss_file, 'w') as f:
    json.dump(loss, f)

# Objective loss
objective_loss_file = kc.RECORDS_FOLDER + '/objective_loss.json'
loss_objective = {group: [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
        for group, tensors in loss_objective.items()}
with open(objective_loss_file, 'w') as f:
    json.dump(loss_objective, f)

# Entropy loss
entropy_loss_file = kc.RECORDS_FOLDER + '/entropy_loss.json'
loss_entropy = {group: [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
        for group, tensors in loss_entropy.items()}
with open(entropy_loss_file, 'w') as f:
    json.dump(loss_entropy, f)

# Critic loss
critic_loss_file = kc.RECORDS_FOLDER + '/critic_loss.json'
loss_critic = {group: [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
        for group, tensors in loss_critic.items()}
with open(critic_loss_file, 'w') as f:
    json.dump(loss_critic, f)"""


############ Plotter
from services import plotter

plotter(params[kc.PLOTTER])

"""plt.figure()
for group in env.group_map.keys():
    rewards = episode_reward_mean_map[group]
    plt.plot(rewards, label=group)

plt.title('Mean Rewards for All Groups')
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.legend()
plt.grid(True)
plt.savefig('rewards.png')

############ Total Loss

plt.figure(figsize=(10, 6))

for agent_id, losses in loss.items():
    losses_np = [loss.detach().cpu().numpy() for loss in losses]
    plt.plot(losses_np, label=f'Agent {agent_id}')

plt.title('Total Loss per Agent')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('total_loss.png')

############ Objective Loss

plt.figure(figsize=(10, 6)) 

for agent_id, losses in loss_objective.items():
    losses_np = [loss.detach().cpu().numpy() for loss in losses]
    plt.plot(losses_np, label=f'Agent {agent_id}')

plt.title('Objective Loss per Agent')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('objective_loss.png')


############ Entropy Loss

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

for agent_id, losses in loss_entropy.items():
    losses_np = [loss.detach().cpu().numpy() for loss in losses]
    plt.plot(losses_np, label=f'Agent {agent_id}')

plt.title('Entropy Loss per Agent')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('entropy_loss.png')


############ Critic Loss

plt.figure(figsize=(10, 6))

for agent_id, losses in loss_critic.items():
    losses_np = [loss.detach().cpu().numpy() for loss in losses]
    plt.plot(losses_np, label=f'Agent {agent_id}')

plt.title('Critic Loss per Agent')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('critic_loss.png')"""
