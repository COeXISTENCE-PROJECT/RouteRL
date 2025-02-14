# %% [markdown]
# # Simulating fleets of automated vehicles (AVs) making routing decisions: Bigger traffic network, IQL algorithm implementation

# %% [markdown]
# > In this notebook, on the `Ingolstadt` network, we simulate **100 human agents** for `1700 days`. After 100 days **20 of the human agents** mutate into automated vehicles (AVs) and use the `IQL` (Independent Q-Learning) algorithm implemented from the `TorchRL` library to learn the optimal route. The AVs are `selfish` and their goal is to maximize their own travel time. Since all AVs have their own reward signal, we model them using independent MARL algorithms. 
# 
# ---

# %% [markdown]
# > The network used.
# > 
# ![Network used](plots_saved/ingolstadt.png)
# 
# ---

# %% [markdown]
# > Tutorial based on [IQL TorchRL Tutorial](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/iql.py).

# %% [markdown]
# #### High-level overview of IQL algorithm
# 
# In IQL a centralized state-action value function is used, Q<sub>tot</sub>, and each agent α learns an individual action-value function Q<sub>α</sub>, independently.
# 
# ---

# %% [markdown]
# #### Imported libraries

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../')))

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

# %% [markdown]
# #### Hyperparameters setting

# %%
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print("device is: ", device)

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
new_machines_after_mutation = 20

# number of episodes the AV training will take
training_episodes = (frames_per_batch / new_machines_after_mutation) * n_iters

env_params = {
    "agent_parameters" : {
        "num_agents" : 100,
        "new_machines_after_mutation": new_machines_after_mutation,
        "human_parameters" : {
            "model" : "w_avg"
        },
    },
    "simulator_parameters" : {
        "network_name" : "ingolstadt"
    },  
    "plotter_parameters" : {
        "phases" : [0, human_learning_episodes, training_episodes + human_learning_episodes],
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

# %% [markdown]
# > Available paths create using the [Janux](https://github.com/COeXISTENCE-PROJECT/JanuX) framework.

# %% [markdown]
# | |  |
# |---------|---------|
# |  ![](plots_saved/0_0.png) |  ![](plots_saved/0_1.png) |
# | ![](plots_saved/1_0.png) | ![](plots_saved/1_1.png) |

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
# #### PettingZoo environment wrapper

# %% [markdown]
# > `TorchRL` enables us to make different groups with different agents. Here, all the AV agents are included in one group.

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
qnet_explore.eval() # set the policy into evaluation mode

num_episodes = 100
for episode in range(num_episodes):
    env.rollout(len(env.machine_agents), policy=qnet_explore)

# %% [markdown]
# > Save the trained policy

# %%
#torch.save(qnet_explore, "trained_policy.pt")

# %% [markdown]
# > Plots of the training process are include in the **\plots** folder.

# %%
env.plot_results()

# %% [markdown]
# > The plots reveal that the introduction of AVs into urban traffic influences human agents' decision-making. This insight highlights the need for research aimed at mitigating potential negative effects of AV introduction, such as increased human travel times, congestion, and subsequent rises in $CO_2$ emissions.

# %% [markdown]
# | |  |
# |---------|---------|
# | **Action shifts of human and AV agents** ![](plots_saved/iql_actions_shifts.png) | **Action shifts of all vehicles in the network** ![](plots_saved/iql_actions.png) |
# | ![](plots_saved/iql_rewards.png) | ![](plots_saved/iql_travel_times.png) |
# 
# 
# <p align="center">
#   <img src="plots_saved/iql_tt_dist.png" width="700" />
# </p>
# 

# %% [markdown]
# > Interrupt the connection with `SUMO`.

# %%
env.stop_simulation()


