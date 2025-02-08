# %% [markdown]
# # MAPPO - IPPO algorithms implementation

# %% [markdown]
# > In this notebook, we implement two state-of-the-art Multi Agent Reinforcement Leaning (MARL) algorithms **Multi-Agent Proximal Policy Optimization [MAPPO](https://arxiv.org/pdf/2103.01955)** and **Independent Proximal Policy Optimization [IPPO](https://arxiv.org/pdf/2011.09533)** in our environment. 
# 
# 
# > Tutorial based on [Multi-Agent Reinforcement Learning (PPO) with TorchRL Tutorial](https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html).

# %% [markdown]
# ### Simulation overview

# %% [markdown]
# > We simulate our environment with an initial population of **200 human agents**. These agents navigate the environment and eventually converge on the fastest path. After this convergence, we will transition **50 of these human agents** into **machine agents**, specifically autonomous vehicles (AVs), which will then employ either the MAPPO or IPPO reinforcement learning algorithms to further refine their learning.

# %% [markdown]
# ![Alt text](../../docs/img/env.png)
# 

# %% [markdown]
# #### Imported libraries

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

import torch
from tqdm import tqdm

from tensordict.nn import TensorDictModule
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
frames_per_batch = 100  # Number of team frames collected per training iteration
n_iters = 50  # Number of sampling and training iterations - the episodes the plotter plots
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 1  # Number of optimization steps per training iteration
minibatch_size = 16  # Size of the mini-batches in each optimization step
lr = 3e-43 # Learning rate
max_grad_norm = 3.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss


policy_network_depth=3
policy_network_num_cells = 64

critic_network_depth=3
critic_network_num_cells = 64

# Human learning phase
human_learning_episodes = 100

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


# %%
reset_td = env.reset()

# %% [markdown]
# #### Policy/Actor network

# %%
share_parameters_policy = False 

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs = env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs = env.action_spec.space.n,
        n_agents = env.n_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=policy_network_depth,
        num_cells=policy_network_num_cells,
        activation_class=torch.nn.Tanh,
    ),
)

# %%
policy_module = TensorDictModule(
    policy_net,
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
share_parameters_critic = True
mappo = False  # IPPO if False

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1, 
    n_agents=env.n_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=critic_network_depth,
    num_cells=critic_network_num_cells,
    activation_class=torch.nn.ReLU,
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
)

# %%
print("Running policy:", policy(env.reset()))

# %%
print("Running value:", critic(env.reset()))

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
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)

# %% [markdown]
# #### PPO loss function

# %%
loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,
)
loss_module.set_keys( 
    reward=env.reward_key,  
    action=env.action_key, 
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
) 

GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

# %% [markdown]
# #### Training loop

# %%
pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []
loss_values = []
loss_entropy = []
loss_objective = []
loss_critic = []

for tensordict_data in collector: ##loops over frame_per_batch

    ## Generate the rollouts
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),  # Adjust index to start from 0
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),  # Adjust index to start from 0
    )

    # Compute GAE for all agents
    with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

    data_view = tensordict_data.reshape(-1)  
    replay_buffer.extend(data_view)

    ## Update the policies of the learning agents
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            ) 

            optim.step()
            optim.zero_grad()

            loss_values.append(loss_value.item())

            loss_entropy.append(loss_vals["loss_entropy"].item())

            loss_objective.append(loss_vals["loss_objective"].item())

            loss_critic.append(loss_vals["loss_critic"].item())


   
    collector.update_policy_weights_()
   
    # Logging
    done = tensordict_data.get(("next", "agents", "done"))  # Get done status for the group

    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)


    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()

# %% [markdown]
# >  Check `\plots` directory to find the plots created from this experiment.

# %%
env.plot_results()

# %%
env.stop_simulation()


