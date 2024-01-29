import pandas as pd
import gymnasium
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional
from keychain import Keychain as kc


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from keychain import Keychain as kc
from simulator import Simulator
from agent import Agent

class TrafficEnvironment(gymnasium.Env):

    def __init__(self, simulation_parameters):
        self.simulator = Simulator(simulation_parameters)
        self.reward_table = []
        print("[SUCCESS] Environment initiated!")
        
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=float)

        self.action_space = Discrete(3)



    def calculate_free_flow_times(self):
        free_flow_cost = self.simulator.calculate_free_flow_times()
        print('[INFO] Free-flow times: ', free_flow_cost)
        return free_flow_cost
        

    def reset(self, seed=None):

        return np.array([0]), {}



    def step(self, joint_action):

        agent_ids = 0

        
        #agent_ids = joint_action[kc.AGENT_ID]
        #print("agent ids is: ", agent_ids)
        sumo_df = self.simulator.run_simulation_iteration(joint_action)

        #### Calculate joint reward based on travel times returned by SUMO
        joint_reward = self.calculate_rewards(sumo_df)
        print("reward is: ", joint_reward)
        print("\n\n")

        #rewards = [joint_reward for i in range(len(sumo_df))]
        #joint_reward = pd.DataFrame({kc.AGENT_ID : agent_ids, kc.REWARD : rewards})
        sample_observation = np.random.uniform(low=0, high=1, size=(1,))


        return sample_observation, joint_reward, True, True, {}


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

    def encode(state, ts_id):
        """Encode the state of the traffic signal into a hashable object."""

        ## state can be the number of vehicles in each path

        # tuples are hashable and can be used as key in python dictionary
        return state