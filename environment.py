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


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

confirm_env_variable(kc.SUMO_HOME, append="tools")
params = get_json(kc.PARAMS_PATH)


from keychain import Keychain as kc
from simulator import Simulator
from agent import Agent

class TrafficEnvironment(ParallelEnv):

    def __init__(self, simulation_parameters, render_mode=None):
        self.simulator = Simulator(simulation_parameters)
        self.reward_table = []
        print("[SUCCESS] Environment initiated!")

        self.possible_agents = ["agent1", "agent2"] #create_agent_objects(params[kc.AGENTS_GENERATION_PARAMETERS], 0)

        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(1,), dtype=float) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: gym.spaces.Discrete(3) for agent in self.possible_agents
        }

        #self.action_space = gym.spaces.Space

        self.render_mode = render_mode

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

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
            a: () for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos



    def step(self, joint_action):

        sumo_df = self.simulator.run_simulation_iteration(joint_action)

        #### Calculate joint reward based on travel times returned by SUMO
        joint_reward = self.calculate_rewards(sumo_df)
        print("reward is: ", joint_reward)
        print("\n\n")

        sample_observation = {
            a: () for a in self.agents
        }

        reward = {
            rew: {joint_reward} for rew in self.agents
        }

        terminated = {
            terminated: {True} for terminated in self.agents
        }

        truncated = {
            truncated: {True} for truncated in self.agents
        }

        info = {
            info: {True} for info in self.agents
        }

        if any(terminated.values()) or all(truncated.values()):
            self.agents = []


        return sample_observation, reward, terminated, truncated, info


    def calculate_rewards(self, sumo_df):
        ### sychronize names
        average_reward = -1 * sumo_df['cost'].mean()
        self.reward_table.append(average_reward)
        return average_reward
    

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
        return gym.spaces.Discrete(3)
