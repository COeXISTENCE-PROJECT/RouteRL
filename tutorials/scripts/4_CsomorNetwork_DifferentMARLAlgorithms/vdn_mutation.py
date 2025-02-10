# %% [markdown]
# # VDN algorithm implementation

# %% [markdown]
# > In this notebook, we implement a state-of-the-art Multi Agent Reinforcement Leaning (MARL) algorithm **[VDN](https://arxiv.org/abs/1706.05296)** in our environment. **VDN** is a deep algorithm for cooperative MARL, particularly suited for situations where agents receive a single, shared reward. Value-decomposition networks are a step towards automatically decomposing complex learning problems into local, more readable learnable sub-problems.
# 
# 
# > Tutorial based on [VDN TorchRL Tutorial](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/qmix_vdn.py).

# %% [markdown]
# <img src="../../docs/img/vdn.png" alt="VDN" width="700"/>
# 
# 
# > Picture taken from VDN [paper](https://arxiv.org/pdf/1706.05296).

# %% [markdown]
# #### High-level overview of VDN algorithm
# 
# The joint action-value function for the system can be additively decomposed into value functions across agents:
# 
# $$
# Q((h^{1}, h^{2}, \ldots, h^{d}), (a^{1}, a^{2}, \ldots, a^{d})) \approx \sum_{i=1}^{d} \tilde{Q}_i(h^{i}, a^{i}),
# $$
# 
# 
# where the $\tilde{Q}_i$ depends only on each agent's local observations.
# 
# **Value-Decomposition** outperforms both centralized and fully independent learning approaches. When combined with additional techniques, it consistently yields agents that significantly surpass their centralized and independent counterparts.
# 

# %% [markdown]
# ### Simulation overview

# %% [markdown]
# > We simulate our environment with an initial population of **200 human agents**. These agents navigate the environment and eventually converge on the fastest path. After this convergence, we will transition **50 of these human agents** into **machine agents**, specifically autonomous vehicles (AVs), which will then employ the QMIX reinforcement learning algorithms to further refine their learning.

# %% [markdown]
# #### Imported libraries

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../')))

import torch
from torch import nn
from tqdm import tqdm

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl._utils import logger as torchrl_logger
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP, VDNMixer
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss

from routerl import TrafficEnvironment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# %% [markdown]
# #### Hyperparameters setting

# %%
# Devices
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("device is: ", device)
vmas_device = device  # The device where the simulator is run

# Sampling
frames_per_batch = 150  # Number of team frames collected per training iteration
n_iters = 100  # Number of sampling and training iterations - the episodes the plotter plots
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 1  # Number of optimization steps per training iteration
minibatch_size = 16  # Size of the mini-batches in each optimization step
lr = 3e-3  # Learning rate
max_grad_norm = 5.0  # Maximum norm for the gradients
memory_size = 1000  # Size of the replay buffer
tau =  0.05
gamma = 0.99  # discount factor

mlp_depth=2
mlp_num_cells=256

eps_greedy_init=0.3
eps_greedy_end=0

mixing_embed_dim = 32

# Human learning phase
human_learning_episodes = 100

# Environment
env_params = {
    "agent_parameters" : {
        "num_agents" : 200,
        "new_machines_after_mutation": 50,
        "human_parameters" : {
            "model" : "w_avg"
        },
    },
    "simulator_parameters" : {
        "network_name" : "csomor"
    },  
    "plotter_parameters" : {
        "phases" : [0, human_learning_episodes],
        "smooth_by" : 50,
    }
}

# %% [markdown]
# #### Environment initialization

# %% [markdown]
# > In this example, the environment initially contains only human agents.

# %%
env = TrafficEnvironment(seed=42, **env_params)
print(env)

# %%
print("Number of total agents is: ", len(env.all_agents), "\n")
print("Agents are: ", env.all_agents, "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

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
# > **Mutation**: a portion of human agents are converted into machine agents (autonomous vehicles). You can adjust the number of agents to be mutated in the <code style="color:white">/params.json</code> file.

# %%
env.mutation()

# %%
print("Number of total agents is: ", len(env.all_agents), "\n")
print("Agents are: ", env.all_agents, "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

# %% [markdown]
# > Create a group that contains all the machine (RL) agents.
# 
# >  **Hint:** the agents aren't completely independent in this example.

# %%
machine_list = []
for machines in env.machine_agents:
    machine_list.append(str(machines.id))
      
group = {'agents': machine_list}

# %% [markdown]
# #### PettingZoo environment wrapper

# %%
env = PettingZooWrapper(
    env=env,
    use_mask=True,
    categorical_actions=True,
    done_on_any = False,
    group_map=group,
    device=device
)

# %% [markdown]
# > Agent group mapping

# %%
print("env.group is: ", env.group_map, "\n\n")

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


# %%
reset_td = env.reset()


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
        share_params=True,
        device=device,
        depth=mlp_depth,
        num_cells=mlp_num_cells,
        activation_class=nn.Tanh,
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
        eps_init=eps_greedy_init,
        eps_end=eps_greedy_end,
        annealing_num_steps=int(total_frames * (1 / 2)),
        action_key=env.action_key,
        spec=env.action_spec,
    ),
)

# %% [markdown]
# #### Mixer
# 
# > `VDNMixer` mixes **the local Q values** of the agents into **a global Q value** by summing them together, according to [VDN paper](https://arxiv.org/pdf/1706.05296).

# %%
mixer = TensorDictModule(
    module=VDNMixer(
        n_agents=env.n_agents,
        device=device,
    ),
    in_keys=[("agents", "chosen_action_value")],
    out_keys=["chosen_action_value"],
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
# #### Qmix loss function

# %% [markdown]
# > `QMixerLoss` mixes *local agent q values* into *a global q value* according to a mixing network and then uses DQN updated on the global value.

# %%
loss_module = QMixerLoss(qnet, mixer, delay_value=True)

loss_module.set_keys(
    action_value=("agents", "action_value"),
    local_value=("agents", "chosen_action_value"),
    global_value="chosen_action_value",
    action=env.action_key,
)

loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
target_net_updater = SoftUpdate(loss_module, eps=1 - tau)

optim = torch.optim.Adam(loss_module.parameters(), lr)


# %% [markdown]
# #### Training loop

# %%
for i, tensordict_data in tqdm(enumerate(collector), total=n_iters, desc="Training"):
    
    ## Generate the rollouts
    tensordict_data.set(
        ("next", "reward"), tensordict_data.get(("next", env.reward_key)).mean(-2)
    )
    del tensordict_data["next", env.reward_key]
    tensordict_data.set(
        ("next", "episode_reward"),
        tensordict_data.get(("next", "agents", "episode_reward")).mean(-2),
    )
    del tensordict_data["next", "agents", "episode_reward"]


    current_frames = tensordict_data.numel()
    total_frames += current_frames
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

            total_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )
            training_tds[-1].set("grad_norm", total_norm.mean())

            optim.step()
            optim.zero_grad()
            target_net_updater.step()

    qnet_explore[1].step(frames=current_frames)  # Update exploration annealing
    collector.update_policy_weights_()

    training_tds = torch.stack(training_tds) 

# %% [markdown]
# >  Check `\plots` directory to find the plots created from this experiment.

# %%
env.plot_results()

# %%
env.stop_simulation()


