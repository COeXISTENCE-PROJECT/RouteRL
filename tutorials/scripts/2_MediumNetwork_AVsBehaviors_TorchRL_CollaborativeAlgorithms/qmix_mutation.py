# %% [markdown]
# # QMIX algorithm implementation

# %% [markdown]
# > In this notebook, we implement a state-of-the-art Multi Agent Reinforcement Leaning (MARL) algorithms **[QMIX](https://arxiv.org/pdf/1803.11485)** in our environment. QMIX is a deep MARL method that allows end-to-end learning of decentralized policies in a centralized setting amd makes efficient use of extra state information. 
# 
# 
# > Tutorial based on [QMIX TorchRL Tutorial](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/qmix_vdn.py).

# %% [markdown]
# ![Qmix](../../docs/img/qmix.png)
# 
# 
# > Picture taken from QMIX [paper](https://arxiv.org/pdf/1803.11485).
# 

# %% [markdown]
# #### High-level overview of QMIX algorithm
# 
# Each agent has its own agent network that represents its individual value function Q<sub>a</sub>. 
# 
# The mixing network is a feed-forward neural network that has as input the agent network outputs and mixes them monotonically. It produces the values of Q<sub>tot</sub>.
# 
# The weights of the mixing network are produced by separate hypernetworks. Each hypernetwork takes the state *s* as input and generated the weights of one layer of the mixing network.

# %% [markdown]
# ### Simulation overview

# %% [markdown]
# > We simulate our environment with an initial population of **100 human agents**. These agents navigate the environment and eventually converge on the fastest path. After this convergence, we will transition **40 of these human agents** into **machine agents**, specifically autonomous vehicles (AVs), which will then employ the QMIX reinforcement learning algorithms to further refine their learning.

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
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer
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

# Sampling
frames_per_batch = 100  # Number of team frames collected per training iteration
n_iters = 300  # Number of sampling and training iterations - the episodes the plotter plots
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 1  # Number of optimization steps per training iteration
minibatch_size = 16  # Size of the mini-batches in each optimization step
lr = 3e-2  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients
memory_size = 1000  # Size of the replay buffer
tau =  0.005
gamma = 0.99  # discount factor

mlp_depth=2
mlp_num_cells=256

eps_greedy_init=0.3
eps_greedy_end=0

mixing_embed_dim = 32

human_learning_episodes = 100


# Environment
env_params = {
    "agent_parameters" : {
        "num_agents" : 100,
        "new_machines_after_mutation": 40,
        "human_parameters" : {
            "model" : "w_avg"
        },
        "machine_parameters" :
        {
            "behavior" : "malicious",
        }
    },
    "simulator_parameters" : {
        "network_name" : "cologne"
    },  
    "plotter_parameters" : {
        "phases" : [0, human_learning_episodes],
        "smooth_by" : 50,
    },
    "path_generation_parameters":
    {
        "number_of_paths" : 3,
        "beta" : -5,
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
    use_mask=True, # Whether to use the mask in the outputs. It is important for AEC environments to mask out non-acting agents.
    categorical_actions=True,
    done_on_any = False, # Whether the environment’s done keys are set by aggregating the agent keys using any() (when True) or all() (when False).
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
        annealing_num_steps=int(total_frames * (1 / 2)), # Number of steps it will take for epsilon to reach the eps_end value
        action_key=env.action_key, # The key where the action can be found in the input tensordict.
        spec=env.action_spec,
    ),
)

# %% [markdown]
# #### Mixer
# 
# > `QMixer` mixes the local Q values of the agents into a global Q value through a monotonic hyper-network whose parameters are obtained from a global state, according to [Qmix paper](https://arxiv.org/pdf/1803.11485).

# %%
mixer = TensorDictModule(
    module=QMixer(
        state_shape=env.observation_spec[
            "agents", "observation"
        ].shape,
        mixing_embed_dim=mixing_embed_dim,
        n_agents=env.n_agents,
        device=device,
    ),
    in_keys=[("agents", "chosen_action_value"), ("agents", "observation")],
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

# %%
loss_module = QMixerLoss(qnet, mixer, delay_value=True)

loss_module.set_keys(
    action_value=("agents", "action_value"),
    local_value=("agents", "chosen_action_value"),
    global_value="chosen_action_value",
    action=env.action_key,
)

loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma) # The value estimator used for the loss computation
target_net_updater = SoftUpdate(loss_module, eps=1 - tau) # Technique used to update the target network

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
# > Testing phase

# %%
num_episodes = 100
for episode in range(num_episodes):
    env.rollout(len(env.machine_agents), policy=qnet_explore)

# %% [markdown]
# >  Check `\plots` directory to find the plots created from this experiment.

# %%
env.plot_results()

# %%
env.stop_simulation()


