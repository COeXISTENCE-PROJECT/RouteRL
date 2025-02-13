# %% [markdown]
# # RouteRL Quickstart
# 
# We simulate a simple network topology where humans and later AVs make routing decisions to maximize their rewards (i.e., minimize travel times) over a sequence of days.
# 
# * For the first 100 days, we model a human-driven system, where drivers update their routing policies using behavioral models to optimize rewards.
# * Each day, we simulate the impact of joint actions using the [`SUMO`](https://eclipse.dev/sumo/) traffic simulator, which returns the reward for each agent.
# * After 100 days, we introduce 10 `Autononmous Vehicles` as `Petting Zoo` agents, allowing them to use any `MARL` algorithm to maximise rewards.
# * Finally, we analyse basic results from the simulation.
#   
# 
# 
# 

# %% [markdown]
# <p align="center">
#   <img src="../../docs/img/two_route_net_1.png" alt="Two-route network" />
#   <img src="../../docs/img/two_route_net_1_2.png" alt="Two-route network" />
# </p>  

# %% [markdown]
# #### Import libraries

# %%
import sys
import os
import pandas as pd
import torch
from torchrl.envs.libs.pettingzoo import PettingZooWrapper


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

from routerl import TrafficEnvironment

# %% [markdown]
# #### Define hyperparameters
# 
# > Further parameters customization can take place by modifying the entries of the `routerl/environment/params.json`. Users can create a dictionary with their preferred adjustments and pass it as an argument to the `TrafficEnvironment` class.

# %%
human_learning_episodes = 100


env_params = {
    "agent_parameters" : {
        "num_agents" : 100,
        "new_machines_after_mutation": 10, # the number of human agents that will mutate to AVs
        "human_parameters" : {
            "model" : "gawron"
        },
        "machine_parameters" :
        {
            "behavior" : "selfish",
        }
    },
    "simulator_parameters" : {
        "network_name" : "two_route_yield"
    },  
    "plotter_parameters" : {
        "phases" : [0, human_learning_episodes], # the number of episodes human learning will take
    },
}

# %% [markdown]
# #### Environment initialization

# %% [markdown]
# In our setup, road networks initially consist of human agents, with AVs introduced later.
# 
# - The `TrafficEnvironment` environment is firstly initialized.
# - The traffic network is instantiated and the paths between designated origin and destination points are determined.
# - The drivers/agents objects are created.

# %%
env = TrafficEnvironment(seed=42, **env_params)

# %% [markdown]
# > Available paths create using the [Janux](https://github.com/COeXISTENCE-PROJECT/JanuX) framework.

# %% [markdown]
# <p >
#   <img src="plots_saved/0_0.png" width="600" />
# </p>  

# %%
print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

# %% [markdown]
# > Reset the environment and the connection with SUMO

# %%
env.start()

# %% [markdown]
# #### Human learning

# %%
for episode in range(human_learning_episodes):
    env.step() # all the human agents execute an action in the environment

# %% [markdown]
# > Average travel time of human agents during their training process.

# %% [markdown]
# <p align="center">
#   <img src="plots_saved/human_learning.png"/>
# </p> 

# %% [markdown]
# > Show the initial `.csv` file saved that contain the information about the agents available in the system.
# 

# %%
df = pd.read_csv("training_records/episodes/ep1.csv")
df


# %% [markdown]
# #### Mutation

# %% [markdown]
# > Mutation: a portion of human agents are converted into machine agents (autonomous vehicles). 

# %%
env.mutation()

# %%
print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

# %%
env.machine_agents

# %% [markdown]
# > In order to employ the `TorchRL` library in our environment we need to use their `PettingZooWrapper` function.

# %%
group = {'agents': [str(machine.id) for machine in env.machine_agents]}

env = PettingZooWrapper(
    env=env,
    use_mask=True,
    categorical_actions=True,
    done_on_any = False,
    group_map=group,
)

# %% [markdown]
# > Use an already trained policy using the Independent Deep Q-Learning algorith.

# %%
qnet_explore = torch.load("trained_policy.pt")

# %% [markdown]
# > Human and AV agents interact with the environment over multiple episodes, with AVs following a random policy as defined in the PettingZoo environment [loop](https://pettingzoo.farama.org/content/basic_usage/).

# %%
num_test_episodes = 100

for episode in range(num_test_episodes): # run rollous in the environment using the already trained policy
    env.rollout(len(env.machine_agents), policy=qnet_explore)

# %% [markdown]
# > Show the first `.csv` file saved after the mutation that contains the information about the agents available in the system after the mutation.

# %%
df = pd.read_csv("training_records/episodes/ep101.csv")
df

# %% [markdown]
# #### Plot results 
# 
# >This will be shown in the `\plots` folder.

# %%
env.plot_results()

# %% [markdown]
# | |  |
# |---------|---------|
# | **Action shifts of human and AV agents** ![](plots_saved/actions_shifts.png) | **Action shifts of all vehicles in the network** ![](plots_saved/actions.png) |
# | ![](plots_saved/rewards.png) | ![](plots_saved/travel_times.png) |
# 
# 
# <p align="center">
#   <img src="plots_saved/tt_dist.png" width="700" />
# </p>
# 

# %% [markdown]
# > Interrupt the connection with `SUMO`.

# %%
env.stop_simulation()


