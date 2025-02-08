# %% [markdown]
# # Independent Q-learning

# %% [markdown]
# > In this notebook, we implement the **[Independent Q learning]()** Multi Agent Reinforcement Leaning (MARL) algorithm in our environment. 
# 
# 
# > Tutorial based on [IQL TorchRL Tutorial](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/iql.py).

# %% [markdown]
# #### High-level overview of IQL algorithm
# 
# In IQL a centralized state-action value function is used, Q<sub>tot</sub>, and each agent α learns an individual action-value function Q<sub>α</sub>, independently.

# %% [markdown]
# ### Simulation overview

# %% [markdown]
# > We simulate our environment with an initial population of **100 human agents**. These agents navigate the environment and eventually converge on the fastest path. After this convergence, we will transition **20 of these human agents** into **machine agents**, specifically autonomous vehicles (AVs), which will then employ the Independent Q learning reinforcement learning algorithm to further refine their learning.

# %% [markdown]
# #### Imported libraries

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

import torch
from tqdm import tqdm

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import SoftUpdate, ValueEstimators, DQNLoss

from routerl import TrafficEnvironment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %%
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("device is: ", device)

# %% [markdown]
# #### Hyperparameters setting

# %%
# Sampling
frames_per_batch = 100  # Number of team frames collected per training iteration
n_iters = 300  # Number of sampling and training iterations - the episodes the plotter plots
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 1  # Number of optimization steps per training iteration 
minibatch_size = 16  # Size of the mini-batches in each optimization step
lr = 3e-3 # Learning rate
max_grad_norm = 5.0  # Maximum norm for the gradients
memory_size = 5_000  # Size of the replay buffer
tau =  0.05
gamma = 0.99  # discount factor
exploration_fraction = 1/3 # Fraction of frames over which the exploration rate is annealed

eps = 1 - tau
eps_init = 0.99
eps_end = 0

mlp_depth = 2
mlp_cells = 32

# Human learning phase
human_learning_episodes = 100

# Environment
env_params = {
    "agent_parameters" : {
        "num_agents" : 100,
        "new_machines_after_mutation": 20,
        "human_parameters" : {
            "model" : "w_avg"
        },
    },
    "simulator_parameters" : {
        "network_name" : "ingolstadt"
    },  
    "plotter_parameters" : {
        "phases" : [0, human_learning_episodes],
        "smooth_by" : 50,
    },
    "path_generation_parameters":
    {
        "number_of_paths" : 5,
        "beta" : -2,
    }
}

# %% [markdown]
# #### Environment initialization

# %% [markdown]
# > In this example, the environment initially contains only human agents.

# %%
env = TrafficEnvironment(seed=42, **env_params)
print(env)

# %% [markdown]
# > Reset the environment and the connection with SUMO

# %%
env.start()
env.reset()

# %% [markdown]
# #### Human learning

# %%
for episode in range(human_learning_episodes):
    env.step()

# %% [markdown]
# #### Mutation

# %% [markdown]
# > **Mutation**: a portion of human agents are converted into machine agents (autonomous vehicles).

# %%
env.mutation()
print(env)

# %% [markdown]
# > Create a group that contains all the machine (RL) agents.
# 

# %% [markdown]
# #### PettingZoo environment wrapper

# %%
group = {'agents': [str(machine.id) for machine in env.machine_agents]}

env = PettingZooWrapper(
    env=env,
    use_mask=True,
    categorical_actions=True,
    done_on_any = False,
    group_map=group,
    device=device
)

# %% [markdown]
# #### Transforms

# %% [markdown]
# Here we instantiate a <code style="color:white">RewardSum</code> transformer that will sum rewards over episode.

# %%
env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

# %% [markdown]
# The <code style="color:white">check_env_specs()</code> function runs a small rollout and compared it output against the environment specs. It will raise an error if the specs aren't properly defined.

# %%
check_env_specs(env)
env.reset()

# %% [markdown]
# #### Policy network

# %% [markdown]
# > Instantiate an `MPL` that can be used in multi-agent contexts.

# %%
net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=env.action_spec.space.n,
        n_agents=env.n_agents,
        centralised=False,
        share_params=False,
        device=device,
        depth=mlp_depth,
        num_cells=mlp_cells,
        activation_class=nn.ReLU,
    )

# %%
module = TensorDictModule(
        net, in_keys=[("agents", "observation")], out_keys=[("agents", "action_value")]
)

# %%
value_module = QValueModule(
    action_value_key=("agents", "action_value"),
    out_keys=[
        env.action_key,
        ("agents", "action_value"),
        ("agents", "chosen_action_value"),
    ],
    spec=env.action_spec,
    action_space=None,
)

qnet = SafeSequential(module, value_module)

# %%
qnet_explore = TensorDictSequential(
    qnet,
    EGreedyModule(
        eps_init=eps_init,
        eps_end=eps_end,
        annealing_num_steps=int(total_frames * exploration_fraction),
        action_key=env.action_key,
        spec=env.action_spec,
    ),
)

# %% [markdown]
# #### Collector

# %%
collector = SyncDataCollector(
        env,
        qnet_explore,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

# %% [markdown]
# #### Replay buffer

# %%
replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(memory_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

# %% [markdown]
# #### DQN loss function

# %%
loss_module = DQNLoss(qnet, delay_value=True)

loss_module.set_keys(
        action_value=("agents", "action_value"),
        action=env.action_key,
        value=("agents", "chosen_action_value"),
        reward=env.reward_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
)

loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
target_net_updater = SoftUpdate(loss_module, eps=eps)

optim = torch.optim.Adam(loss_module.parameters(), lr)

# %% [markdown]
# #### Training loop

# %%
for i, tensordict_data in tqdm(enumerate(collector), total=n_iters, desc="Training"):
    
    current_frames = tensordict_data.numel()
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)
    
    training_tds = []

    ## Update the policies of the learning agents
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)
            training_tds.append(loss_vals.detach())

            loss_value = loss_vals["loss"]
            loss_value.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            training_tds[-1].set("grad_norm", total_norm.mean())

            optim.step()
            optim.zero_grad()
        target_net_updater.step()

    qnet_explore[1].step(frames=current_frames)  # Update exploration annealing
    collector.update_policy_weights_()
    
    training_tds = torch.stack(training_tds) 

collector.shutdown()


# %% [markdown]
# > Testing phase

# %%
num_episodes = 100
for episode in range(num_episodes):
    env.rollout(len(env.machine_agents), policy=qnet_explore)

# %% [markdown]
# > Plots of the training process are include in the **\plots** folder.

# %%
env.plot_results()

# %%
env.stop_simulation()


