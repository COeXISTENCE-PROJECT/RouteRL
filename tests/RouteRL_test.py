# Introduction to the TrafficEnvironment

#TrafficEnvironment integrates multi-agent reinforcement learning (MARL) with 
# a microscopic traffic simulation tool to explore the potential of MARL in optimizing urban route choice. 
# The aim of the framework is to simulate the coexistence of human drivers and Automated Vehicles (AVs) in city networks.


import sys
import os
from tqdm import tqdm
import numpy as np

from routerl import TrafficEnvironment


#### Define hyperparameters

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

#### Environment initialization

# In this example, the environment initially contains only human agents.


# In our setup, road networks initially consist of human agents, with AVs introduced later. However, RouteRL is flexible and can operate with only AV agents, only human agents, or a mix of both.

env = TrafficEnvironment(seed=42, **env_params)

print(env)


print("Number of total agents is: ", len(env.all_agents), "\n")
print("Agents are: ", env.all_agents, "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")


# Reset the environment and the connection with SUMO
env.start()

#### Human learning
for episode in range(human_learning_episodes):
    
    env.step()

#### Mutation
# Mutation: a portion of human agents are converted into machine agents (autonomous vehicles). 
env.mutation()


print("Number of total agents is: ", len(env.all_agents), "\n")
print("Agents are: ", env.all_agents, "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")


print(env.machine_agents)

# Human and AV agents interact with the environment over multiple episodes, 
# with AVs following a random policy as defined in the PettingZoo environment loop.
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
