# %% [markdown]
# # Introduction to the TrafficEnvironment

# %% [markdown]
# > TrafficEnvironment integrates multi-agent reinforcement learning (MARL) with a microscopic traffic simulation tool to explore the potential of MARL in optimizing urban route choice. The aim of the framework is to simulate the coexistence of human drivers and Automated Vehicles (AVs) in city networks.
# 
# > We use [SUMO](https://sumo.dlr.de/docs/index.html), an open-source, microscopic and continuous traffic simulation.
# 
# ## Related work
# 
# > Some methods have utilized MARL for optimal route choice (Thomasini et al. [2023](https://alaworkshop2023.github.io/papers/ALA2023_paper_69.pdf/)). These approaches
# are typically based on macroscopic traffic simulations, which model relationships among traffic
# flow characteristics such as density, flow, and mean speed of a traffic stream. In contrast, our
# problem employs a microscopic model, which focuses on interactions between individual vehicles.
# 
# > Additionally, a method proposed by (Tavares and Bazzan [2012](https://www.researchgate.net/publication/235219033_Reinforcement_learning_for_route_choice_in_an_abstract_traffic_scenario)) addresses optimal route choice at the microscopic level, where rewards are generated through a predefined function. In contrast, in our approach, rewards are provided dynamically by a continuous traffic simulator.

# %% [markdown]
# #### Import libraries

# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

from routerl import TrafficEnvironment

# %% [markdown]
# #### Define hyperparameters
# 
# > Further adjustments can be made by modifying the parameters in <code style="color:white">routerl/environment/params.json</code>

# %%
human_learning_episodes = 100


env_params = {
    "agent_parameters" : {
        "num_agents" : 100,
        "new_machines_after_mutation": 10,
        "human_parameters" : {
            "model" : "w_avg"
        },
        "machine_parameters" :
        {
            "behavior" : "malicious",
        }
    },
    "simulator_parameters" : {
        "network_name" : "two_route_yield"
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
# 
# 
# > In our setup, road networks initially consist of human agents, with AVs introduced later. However, RouteRL is flexible and can operate with only AV agents, only human agents, or a mix of both.

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

# %% [markdown]
# #### Human learning

# %%
env.start()

for episode in range(human_learning_episodes):
    
    env.step()


# %% [markdown]
# #### Mutation

# %% [markdown]
# > Mutation: a portion of human agents are converted into machine agents (autonomous vehicles). 

# %%
env.mutation()

# %%
print("Number of total agents is: ", len(env.all_agents), "\n")
print("Agents are: ", env.all_agents, "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

# %%
env.machine_agents

# %% [markdown]
# > Human and AV agents interact with the environment over multiple episodes, with AVs following a random policy as defined in the PettingZoo environment [loop](https://pettingzoo.farama.org/content/basic_usage/).

# %%
episodes = 1

for episode in range(episodes):
    print(f"\nStarting episode {episode + 1}")
    env.reset()
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # Policy action or random sampling
            action = env.action_space(agent).sample()
        print(f"Agent {agent} takes action: {action}")
        
        env.step(action)
        print(f"Agent {agent} has stepped, environment updated.\n")


# %%
env.stop_simulation()

# %% [markdown]
# <code style="color:white">agent_iter(max_iter=2**63)</code> returns an iterator that yields the current agent of the environment. It terminates when all agents in the environment are done or when max_iter (steps have been executed).
# 
# <code style="color:white">last(observe=True)</code> returns observation, reward, done, and info for the agent currently able to act. The returned reward is the cumulative reward that the agent has received since it last acted. If observe is set to False, the observation will not be computed, and None will be returned in its place. Note that a single agent being done does not imply the environment is done.
# 
# <code style="color:white">reset()</code> resets the environment and sets it up for use when called the first time. This method must be called before any other method.
# 
# <code style="color:white">step(action)</code> takes and executes the action of the agent in the environment, automatically switches control to the next agent.


