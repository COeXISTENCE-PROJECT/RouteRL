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
from torch.distributions.categorical import Categorical

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)


from keychain import Keychain as kc
from simulator import Simulator


## link https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/
class TrafficEnvironment(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "TrafficEnvironment",
        "is_parallelizable": True,
    }

    def __init__(self, simulation_parameters, render_mode=None):
        self.simulator = Simulator(simulation_parameters)
        self.reward_table = []
        print("[SUCCESS] Environment initiated!")

        #self.possible_agents = ["1", "2"] 
        self.possible_agents = [str(i) for i in range(1, 601)]

        self.agents = self.possible_agents

        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(1,), dtype=float) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: gym.spaces.Discrete(3) for agent in self.possible_agents
        }

        ### Create start_time table
        step_size = 6

        self.start_times = [i * step_size for i in range(len(self.possible_agents))]

        #self.start_times = [0, 6]

        self.render_mode = render_mode

        self.od_pairs = []
        number_of_agents = len(self.possible_agents)

        for i in range(number_of_agents):
            if i < number_of_agents // 2:
                self.od_pairs.append("0_0")
            else:
                self.od_pairs.append("1_1")



    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return Box(low=0, high=1, shape=(1,), dtype=float).sample() 

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        print("Reward table is: ", self.reward_table)
        self.plot_rewards()


    def calculate_free_flow_times(self):
        free_flow_cost = self.simulator.calculate_free_flow_times()
        print('[INFO] Free-flow times: ', free_flow_cost)
        return free_flow_cost
        

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)

        observations = {
            a: Box(low=0, high=1, shape=(1,), dtype=float).sample() for a in self.agents
        }

        infos = {a: {}  for a in self.agents}

        return observations, infos



    def step(self, joint_action):

        if not joint_action:
            self.agents = []
            return {}, {}, {}, {}, {}
        

        ### Interact with SUMO to get travel times
        sumo_df = self.simulator.run_simulation_iteration(joint_action, self.start_times, self.od_pairs)
        
        costs = sumo_df['cost'].values


        ### Individual reward to each agent
        rewards = {}

        # each agent tries to minimize each one travel time
        i = 0
        for agent_name in self.possible_agents:
            rewards[agent_name] = -1 * costs[i]
            

            if(i == 500):
                self.reward_table.append(-1 * costs[i])

            i = i + 1

        #print(rewards)

        ### Joint reward for all agents
        """joint_reward = self.calculate_rewards(sumo_df)

        print("\n\njoint_reward is: ", joint_reward, "\n\n")

        for agent_name in self.possible_agents:
            rewards[agent_name] = joint_reward"""

        #print("\n\n", rewards, "\n\n")

        ### Return variables
        sample_observation = {
            a: (Box(low=0, high=1, shape=(1,), dtype=float).sample()) for a in self.possible_agents
        }

        terminated = {
            terminated: True for terminated in self.possible_agents
        }

        truncated = {
            truncated: 1 for truncated in self.possible_agents
        }

        info = {a: {} for a in self.agents} 

        if any(terminated.values()) or all(truncated.values()):
            self.agents = []

        return sample_observation, rewards, terminated, truncated, info
    

    def plot_rewards(self):
        plt.figure(figsize=(10, 6)) 
        plt.plot(self.reward_table, color='blue', linestyle='-')  
        plt.xlabel('Episode', fontsize=12) 
        plt.ylabel('Reward', fontsize=12) 
        plt.title('Reward Table Over Episodes', fontsize=14)  
        plt.grid(True, linestyle='--', alpha=0.7)  
        plt.tight_layout() 
        plt.show()

    def calculate_rewards(self, sumo_df):
        average_reward = -1 * sumo_df['cost'].mean()
        self.reward_table.append(average_reward)
        return average_reward


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(1,), dtype=float)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)
