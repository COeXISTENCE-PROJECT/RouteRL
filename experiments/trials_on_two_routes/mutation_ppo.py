import matplotlib.pyplot as plt
import os
import pandas as pd
from tensordict.nn import TensorDictModule, TensorDictSequential
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
frames_per_batch = 200  # Number of team frames collected per training iteration
n_iters = 20  # Number of sampling and training iterations - the episodes the plotter plots
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
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

############ Environment creation

params = get_params(kc.PARAMS_PATH)

env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS], params[kc.PHASE])

env.start()
env.reset()

############ Human learning

num_episodes = 200

for episode in range(num_episodes):
    env.step()


############ Mutation

env.mutation()

############ Machine learning

env = PettingZooWrapper(
    env=env,
    use_mask=True,
    group_map=None,
    categorical_actions=True,
    done_on_any = False,
    device=device
)

env = TransformedEnv(
    env,
    RewardSum(
        in_keys=env.reward_keys,
        reset_keys=["_reset"] * len(env.group_map.keys()),
    ),
    device = device
)

check_env_specs(env)

reset_td = env.reset()


############ Policy network

policy_modules = {}
for group, agents in env.group_map.items():
    share_parameters_policy = False  # Can change this based on the group

    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[group, "observation"].shape[
            -1
        ],  # n_obs_per_agent
        n_agent_outputs= env.full_action_spec[group, "action"].space.n,  # n_actions_per_agents
        n_agents=len(agents),  # Number of agents in the group
        centralised=False,  # the policies are decentralised (i.e., each agent will act from its local observation)
        share_params=share_parameters_policy,
        device=device,
        depth=3,
        num_cells=64,
        activation_class=torch.nn.Tanh,
    )

    # Wrap the neural network in a :class:`~tensordict.nn.TensorDictModule`.
    # This is simply a module that will read the ``in_keys`` from a tensordict, feed them to the
    # neural networks, and write the
    # outputs in-place at the ``out_keys``.

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "logits")],
    )  # We just name the input and output that the network will read and write to the input tensordict
    policy_modules[group] = policy_module

policies = {}
for group, _agents in env.group_map.items():
    policy = ProbabilisticActor(
        module=policy_modules[group],
        spec=env.full_action_spec[group, "action"],
        in_keys=[(group, "logits")],
        out_keys=[(group, "action")],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=(group, "sample_log_prob"),
    )
    policies[group] = policy



############ Critic network
critic_modules = {}
for group, agents in env.group_map.items():
    share_parameters_critic = False
    mappo = False  # IPPO if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],
        n_agent_outputs=1, 
        n_agents=len(agents),
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=4,
        num_cells=64,
        activation_class=torch.nn.Tanh,
    )

    critic_module = TensorDictModule(
        module=critic_net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "state_value")],
    )
    critic_modules[group] = critic_module


reset_td = env.reset()
for group, _agents in env.group_map.items():
    critic_modules[group](policies[group](reset_td))

policy = TensorDictSequential(*policies.values())

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

replay_buffers = {}
for group, _agents in env.group_map.items():
    replay_buffers[group] = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,  # We will sample minibatches of this size
    )



############ PPO loss function

loss_modules = {}
losses = {}
optimizers = {}

for group, _agents in env.group_map.items():
    loss_module = ClipPPOLoss(
        actor_network=policies[group],
        critic_network=critic_modules[group],
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys( 
        reward=(group, "reward"),  
        action=(group, "action"), 
        sample_log_prob=(group, "sample_log_prob"),
        value=(group, "state_value"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        advantage=(group, "advantage")
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
    ) 

    GAE = loss_module.value_estimator

    losses[group] = loss_module

    optimizer = torch.optim.Adam(loss_module.parameters(), lr)
    loss_modules[group] = (loss_module, optimizer)

    optimizers[group] = optimizer


group = next(iter(env.group_map))
loss_module, optimizer = loss_modules[group]



############ Training loop

pbar = tqdm(
    total=n_iters,
    desc=", ".join(
        [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
    ),
)
episode_reward_mean_map = {group: [] for group in env.group_map.keys()}

loss = {group: [] for group in env.group_map.keys()}
loss_objective = {group: [] for group in env.group_map.keys()}
loss_critic = {group: [] for group in env.group_map.keys()}
loss_entropy = {group: [] for group in env.group_map.keys()}

for tensordict_data in collector: ##loops over frame_per_batch

    # Update done and terminated for both agents
    for group, _agents in env.group_map.items():
        tensordict_data.set(
            ("next", group, "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", group, "reward"))),  # Adjust index to start from 0
        )
        tensordict_data.set(
            ("next", group, "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", group, "reward"))),  # Adjust index to start from 0
        )

    # Compute GAE for both agents
    with torch.no_grad():
        for group, _agents in env.group_map.items():
            module = GAE(
                tensordict_data,
                params=loss_modules[group][0].critic_network_params,
                target_params=loss_modules[group][0].target_critic_network_params,
            )

    # Flatten and extend data for both agents
    for group, _agents in env.group_map.items():
        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffers[group].extend(data_view)

    for epoch in range(num_epochs):
        for group, _agents in env.group_map.items():
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffers[group].sample()
                #print("Inside inner loop", subdata, replay_buffers, "\n\n")
                loss_vals = losses[group](subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                if torch.isnan(loss_value).any():
                    raise ValueError("NaNs in the loss")

                loss[group].append(loss_value)
                loss_objective[group].append(loss_vals["loss_objective"])
                loss_critic[group].append(loss_vals["loss_critic"])
                loss_entropy[group].append(loss_vals["loss_entropy"])

                torch.nn.utils.clip_grad_norm_(
                    losses[group].parameters(), max_grad_norm
                )  # Optional

                optimizers[group].step()
                optimizers[group].zero_grad()

    # Update policy weights for both agents
    #for group, _agents in env.group_map.items():
    collector.update_policy_weights_()
   

    for group, _agents in env.group_map.items():
        done = tensordict_data.get(("next", group, "done"))  # Get done status for the group

        episode_reward_mean = (
            tensordict_data.get(("next", group, "reward"))[
                tensordict_data.get(("next", group, "done"))
            ]
            .mean()
            .item()
        )

        episode_reward_mean_map[group].append(episode_reward_mean)

    pbar.set_description(
        ", ".join(
            [
                f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                for group in env.group_map.keys()
            ]
        ),
        refresh=False,
    )
    pbar.update()


############ Save
# Mean episode reward
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
    json.dump(loss_critic, f)


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
