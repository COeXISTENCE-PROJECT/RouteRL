# %% [markdown]
# # SAC algorithm implementation

# %% [markdown]
# > In this notebook, we implement the multi-agent versions of the **[SAC](https://arxiv.org/pdf/1801.01290)** algorithm, **Independent SAC (ISAC)** and **Multi-Agent SAC (MASAC)** in our environment. SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy RL framework. 
# 
# 
# > Tutorial based on [SAC TorchRL Tutorial](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/sac.py).

# %% [markdown]
# ### Simulation overview

# %% [markdown]
# > We simulate our environment with an initial population of **100 human agents**. These agents navigate the environment and eventually converge on the fastest path. After this convergence, we will transition **23 of these human agents** into **machine agents**, specifically autonomous vehicles (AVs), which will then employ the QMIX reinforcement learning algorithms to further refine their learning.

# %% [markdown]
# #### Imported libraries

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

import torch
from tqdm import tqdm

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torch.distributions import Categorical, OneHotCategorical
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import DiscreteSACLoss, SACLoss, SoftUpdate, ValueEstimators

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
memory_size =  1000  # Size of the replay buffer
tau =  0.005
gamma = 0.99  # discount factor

policy_net_depth=3
policy_net_num_cells=64

critic_net_depth=2
critic_net_num_cells=64

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

# %% [markdown]
# > We can append any TorchRL transform we need to our environment. 

# %%
env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

# %% [markdown]
# The <code style="color:white">check_env_specs()</code> function runs a small rollout and compared it output against the environment specs. It will raise an error if the specs aren't properly defined.

# %%
reset_td = env.reset()

# %% [markdown]
# #### Policy network

# %% [markdown]
# > Instantiate an `MPL` that can be used in multi-agent contexts.

# %%
share_parameters_policy = False 

actor_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs = env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs = env.action_spec.space.n,
        n_agents = env.n_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=policy_net_depth,
        num_cells=policy_net_num_cells,
        activation_class=torch.nn.Tanh,
    ),
)

# %%
policy_module = TensorDictModule(
    actor_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "logits")],
) 

# %%
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=[("agents", "logits")],
    out_keys=[env.action_key],
    distribution_class=Categorical,
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob"),
)

# %% [markdown]
# #### Critic network

# %% [markdown]
# > The critic reads the observations and returns the corresponding value estimates.

# %%
centralised_critic = False
shared_parameters = False

module = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=env.action_spec.space.n,
            n_agents=env.n_agents,
            centralised=centralised_critic,
            share_params=shared_parameters,
            device=device,
            depth=critic_net_depth,
            num_cells=critic_net_num_cells,
            activation_class=nn.Tanh,
        )

# %%
value_module = ValueOperator(
            module=module,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "action_value")],
        )

# %% [markdown]
# #### Collector

# %%
collector = SyncDataCollector(
    env,
    policy,
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
# #### SAC loss function

# %%
loss_module = DiscreteSACLoss(
            actor_network=policy,
            qvalue_network=value_module,
            delay_qvalue=True,
            num_actions=env.action_spec.space.n,
            action_space=env.action_spec,
        )

loss_module.set_keys(
    action_value=("agents", "action_value"),
    action=env.action_key,
    reward=env.reward_key,
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)


loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
target_net_updater = SoftUpdate(loss_module, eps=1 - tau)

optim = torch.optim.Adam(loss_module.parameters(), lr)

# %% [markdown]
# #### Training loop

# %%
for i, tensordict_data in tqdm(enumerate(collector), total=n_iters, desc="Training"):


    current_frames = tensordict_data.numel()
    total_frames += current_frames
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    training_tds = []
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)
            training_tds.append(loss_vals.detach())

            loss_value = (
                loss_vals["loss_actor"]
                + loss_vals["loss_alpha"]
                + loss_vals["loss_qvalue"]
            )

            loss_value.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )
            training_tds[-1].set("grad_norm", total_norm.mean())

            optim.step()
            optim.zero_grad()
            target_net_updater.step()

    collector.update_policy_weights_()

    training_tds = torch.stack(training_tds)

# %% [markdown]
# > Testing phase

# %%
num_episodes = 100
for episode in range(num_episodes):
    env.rollout(len(env.machine_agents), policy=policy)

# %% [markdown]
# >  Check `\plots` directory to find the plots created from this experiment.

# %%
env.plot_results()

# %%
env.stop_simulation()


