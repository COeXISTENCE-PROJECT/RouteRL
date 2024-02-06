import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from prettytable import PrettyTable

from keychain import Keychain as kc
from simulator import Simulator

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class TrafficEnvironment(gym.Env):

    def __init__(self, simulation_parameters, agents_data_path):
        self.simulator = Simulator(simulation_parameters)
        self.reward_table = list()
        self.route_flows = list()
        self.all_selected_routes = set()
        self.agents_data_path=agents_data_path
        print("[SUCCESS] Environment initiated!")


    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self.print_free_flow_times(free_flow_times)
        return free_flow_times
        

    def reset(self):
        return None


    def step(self, joint_action):

        agent_ids = joint_action[kc.AGENT_ID]
        sumo_df= self.simulator.run_simulation_iteration(joint_action)

        selected_routes = self.simulator.selected_routes
        self.all_selected_routes.update(selected_routes)
        selected_routes_df = pd.DataFrame(selected_routes, columns=['route_id']).value_counts().values
        self.route_flows.append(selected_routes_df)

        #### Calculate joint reward based on travel times returned by SUMO
        joint_reward = self.calculate_rewards(sumo_df)

        #rewards = [joint_reward for i in range(len(agent_ids))]
        rewards = joint_reward.values.tolist()
        joint_reward = pd.DataFrame({kc.AGENT_ID : agent_ids, kc.REWARD : rewards})

        return joint_reward, None, True


    def calculate_rewards(self, sumo_df):
        ### sychronize names
        agent_data=pd.read_csv(self.agents_data_path)
        real_reward = pd.merge(sumo_df,agent_data,left_on='car_id',right_on='id',how='right')
        real_reward = real_reward.fillna(100)
        real_reward = real_reward.cost
        average_reward = real_reward.mean() 
        self.reward_table.append(average_reward)
        return real_reward


    def print_free_flow_times(self, free_flow_times):
        table = PrettyTable()
        table.field_names = ["Origin", "Destination", "Index", "FF Time"]

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                table.add_row([od[0], od[1], idx, "%.3f"%time])
            table.add_row(["----", "----", "----", "----"])

        print("------ Free flow travel times ------")
        print(table)


    def plot_one_agent(self):
        _, axs = plt.subplots(2, 1)

        flows_df = pd.DataFrame(self.route_flows, columns = list(self.all_selected_routes))

        axs[0].plot(flows_df)
        axs[0].legend(flows_df.columns)

        learning_data = pd.read_csv(kc.ONE_AGENT_EXPERIENCE_LOG_PATH).cost_table.str.split(',',expand=True)

        for i in range(len(learning_data.columns)):
            axs[1].plot(learning_data[i])
        axs[1].legend(learning_data.columns)

        plt.show()


    def plot_rewards(self):

        plt.plot(self.reward_table)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Table Over Episodes')
        plt.show()