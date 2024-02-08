import os
import pandas as pd
import traci

from prettytable import PrettyTable

from keychain import Keychain as kc
from simulator import Simulator

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class TrafficEnvironment:

    def __init__(self, environment_params, simulation_params):
        self.transport_penalty = environment_params[kc.TRANSPORT_PENALTY]
        self.agents_data_path = environment_params[kc.AGENTS_DATA_PATH]
        
        self.simulator = Simulator(simulation_params)
        print("[SUCCESS] Environment initiated!")


    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self.print_free_flow_times(free_flow_times)
        return free_flow_times
        

    def reset(self):
        sumo_observation = self.simulator.reset()
        state = None
        return state


    def step(self, joint_action):
        sumo_df = self.simulator.run_simulation_iteration(joint_action)
        joint_reward = self.calculate_rewards(sumo_df)
        next_state, done = None, True
        return joint_reward, next_state, done


    def calculate_rewards(self, sumo_df):
        # Fill up for transports
        agent_data = pd.read_csv(self.agents_data_path)
        filled_reward = pd.merge(sumo_df, agent_data, left_on=kc.AGENT_ID, right_on=kc.AGENT_ID, how='right')
        filled_reward = filled_reward.fillna(self.transport_penalty)
        # Calculate reward from cost (skipped)
        # Turn cost column to reward, drop everything but id and reward
        filled_reward = filled_reward.rename(columns={kc.TRAVEL_TIME : kc.REWARD})
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