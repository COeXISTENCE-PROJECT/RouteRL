import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
import numpy as np
import matplotlib.pyplot as plt
import os
from prettytable import PrettyTable
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from keychain import Keychain as kc
from simulator import Simulator
from agent import Agent

class TrafficEnvironment(gym.Env):

    def __init__(self, simulation_parameters):
        self.simulator = Simulator(simulation_parameters)
        self.reward_table = []
        print("[SUCCESS] Environment initiated!")



    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self.print_free_flow_times(free_flow_times)
        return free_flow_times
        

    def reset(self):
        return None



    def step(self, joint_action):

        agent_ids = joint_action[kc.AGENT_ID]
        sumo_df = self.simulator.run_simulation_iteration(joint_action)

        #### Calculate joint reward based on travel times returned by SUMO
        joint_reward = self.calculate_rewards(sumo_df)

        rewards = [joint_reward for i in range(len(agent_ids))]
        joint_reward = pd.DataFrame({kc.AGENT_ID : agent_ids, kc.REWARD : rewards})

        return joint_reward, None, True


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



    def print_free_flow_times(self, free_flow_times):
        table = PrettyTable()
        table.field_names = ["Origin", "Destination", "Index", "FF Time"]

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                table.add_row([od[0], od[1], idx, "%.3f"%time])
            table.add_row(["----", "----", "----", "----"])

        print("------ Free flow travel times ------")
        print(table)