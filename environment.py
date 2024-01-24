import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


from keychain import Keychain as kc
from simulator import Simulator
from agent import Agent

class TrafficEnvironment:

    """
    To be implemented
    """

    def __init__(self, simulation_parameters): # get params for simulator
        # Initialize network
        # Create demand
        # Create paths
        # Calculate free flows
        # done
        self.simulator = Simulator(simulation_parameters)  # pass params for simulator, and only the number of agents
        self.reward_table = []

        print("[SUCCESS] Environment initiated!")



    def calculate_free_flow_times(self):
        free_flow_cost = self.simulator.calculate_free_flow_times()
        print('[INFO] Free-flow times: ', free_flow_cost)
        return free_flow_cost
        
        

    def reset(self):
        return None



    def step(self, joint_action):
        agent_ids = joint_action[kc.AGENT_ID]

        ####
        #### Feed agents actions to SUMO and get travel times
        ####
        simulation_length = 3600

        sumo_df = self.simulator.run_simulation_iteration(simulation_length, joint_action)

        #### Calculate joint reward based on travel times returned by SUMO
        joint_reward = self.calculate_rewards(sumo_df)

        rewards = [joint_reward for i in range(len(agent_ids))]
        joint_reward = pd.DataFrame({kc.AGENT_ID : agent_ids, kc.REWARD : rewards})

        return joint_reward, None, True


    def calculate_rewards(self, sumo_df):
        average_travel_time = -1 * sumo_df['cost'].mean()
        self.reward_table.append(average_travel_time)

        return average_travel_time
    
    def plot_rewards(self):

        # Plotting
        plt.plot(self.reward_table)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Table Over Episodes')
        plt.show()