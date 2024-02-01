from copy import copy
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional
from keychain import Keychain as kc
from pettingzoo import ParallelEnv
from services import create_agent_objects
from services import confirm_env_variable
from services import get_json
import functools
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.vector_env import VectorEnv
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)


from keychain import Keychain as kc
from simulator import Simulator
from agent import Agent


## link https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/
class TrafficEnvironment(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "traffic_environment",
        "is_parallelizable": True,
    }

    def __init__(self, simulation_parameters, render_mode=None):
        self.simulator = Simulator(simulation_parameters)
        self.reward_table = []
        print("[SUCCESS] Environment initiated!")

        self.possible_agents = ["agent1", "agent2"] #create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], 0)
        self.agents = self.possible_agents

        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(1,), dtype=float) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: gym.spaces.Discrete(3) for agent in self.possible_agents
        }

        #self.action_space = gym.spaces.Space
        self.agent_selection = 0 ## in the pistoball.py it is a number

        self.render_mode = render_mode

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        #print("\n\n\n\nself.observation_spaces[agent]", self.observation_spaces.get(agent), "\n\n\n\n\n")
        return 0 #np.array(self.observation_spaces[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass


    def calculate_free_flow_times(self):
        free_flow_cost = self.simulator.calculate_free_flow_times()
        print('[INFO] Free-flow times: ', free_flow_cost)
        return free_flow_cost
        

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)

        observations = {
            a: (Box(low=0, high=1, shape=(1,), dtype=float).sample()) for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {}  for a in self.agents}

        return [observations, infos] # for the parallel_api_test [observations, infos]



    def step(self, joint_action):

        self.agent_selection = 0
        
        sumo_df = self.simulator.run_simulation_iteration(joint_action)
        costs = sumo_df['cost'].values

        print("costs is: ", costs, "\n\n")

        rewards = {}

        # Iterate over the possible_agents list
        for agent_name in self.possible_agents:
            # Assign the corresponding reward for each agent
            # If you want to assign the same reward to all agents, you can replace `costs[i]` with a single value
            rewards[agent_name] = costs[len(rewards)] if len(rewards) < len(costs) else None

        print("typeof rewards", type(rewards), "\n\n")

        sample_observation = {
            a: (Box(low=0, high=1, shape=(1,), dtype=float).sample()) for a in self.possible_agents
        }

        terminated = {
            terminated: {True} for terminated in self.possible_agents
        }

        truncated = {
            truncated: {True} for truncated in self.possible_agents
        }

        info = {a: {} for a in self.agents} 

        if any(terminated.values()) or all(truncated.values()):
            self.agents = []

        print("\n\nbefore the return\n\n")

        return [sample_observation, rewards, terminated, truncated, info] ##for rllib not the truncated
    

    def plot_rewards(self):
        plt.plot(self.reward_table)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Table Over Episodes')
        plt.show()

    def create_agents(self, agents):
        self.possible_agents = agents

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(1,), dtype=float)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)
