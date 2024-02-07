import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import traci

from prettytable import PrettyTable

from keychain import Keychain as kc
from simulator import Simulator
from services import make_dir, df_to_prettytable

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class TrafficEnvironment(gym.Env):

    def __init__(self, simulation_parameters, agents_data_path):
        self.simulator = Simulator(simulation_parameters)
        self.params = simulation_parameters

        self.transport_penalty = self.params[kc.TRANSPORT_PENALTY]
        self.sumo_config_path = self.params[kc.SUMO_CONFIG_PATH]

        self.agents_data_path=agents_data_path
        print("[SUCCESS] Environment initiated!")


    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self.print_free_flow_times(free_flow_times)
        return free_flow_times
        

    def reset(self):
        traci.load(['-c', self.sumo_config_path])
        return None


    def step(self, joint_action):

        sumo_df = self.simulator.run_simulation_iteration(joint_action)
        joint_reward = self.calculate_rewards(sumo_df)

        #agent_ids = joint_action[kc.AGENT_ID]
        #rewards = joint_reward.values.tolist()
        #joint_reward = pd.DataFrame({kc.AGENT_ID : agent_ids, kc.REWARD : rewards})

        return joint_reward, None, True


    def calculate_rewards(self, sumo_df):
        # Fill up for transports
        agent_data = pd.read_csv(self.agents_data_path)
        filled_reward = pd.merge(sumo_df, agent_data, left_on=kc.AGENT_ID, right_on=kc.AGENT_ID, how='right')
        filled_reward = filled_reward.fillna(self.transport_penalty)
        # Calculate reward from cost (skipped)
        # Turn cost column to reward, drop everything but id and reward
        filled_reward = filled_reward.rename(columns={kc.COST : kc.REWARD})
        filled_reward = filled_reward[[kc.AGENT_ID, kc.REWARD]]
        return filled_reward


    def print_free_flow_times(self, free_flow_times):
        table = PrettyTable()
        table.field_names = ["Origin", "Destination", "Index", "FF Time"]

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                table.add_row([od[0], od[1], idx, "%.3f"%time])
            table.add_row(["----", "----", "----", "----"])

        print("------ Free flow travel times ------")
        print(table)