from prettytable import PrettyTable

from keychain import Keychain as kc
from services import Simulator

class TrafficEnvironment:

    def __init__(self, environment_params, simulation_params):
        self.simulator = Simulator(simulation_params)
        print("[SUCCESS] Environment initiated!")


    def start(self):
        self.simulator.start_sumo()
        state = None
        return state

    def stop(self):
        self.simulator.stop_sumo()
        state = None
        return state

    def reset(self):
        self.simulator.reset_sumo()
        state = None
        return state


    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self.print_free_flow_times(free_flow_times)
        return free_flow_times


    def step(self, joint_action):
        sumo_df = self.simulator.run_simulation_iteration(joint_action)
        joint_reward = self.calculate_rewards(sumo_df)
        next_state, done = None, True
        return joint_reward, next_state, done


    def calculate_rewards(self, sumo_df):
        # Calculate reward from cost (skipped)
        # Turn cost column to reward, drop everything but id and reward
        reward_df = sumo_df.rename(columns={kc.TRAVEL_TIME : kc.REWARD})
        reward_df = reward_df[[kc.AGENT_ID, kc.REWARD]]
        return reward_df


    def get_last_sim_duration(self):
        return self.simulator.get_last_sim_duration()


    def print_free_flow_times(self, free_flow_times):
        table = PrettyTable()
        table.field_names = ["Origin", "Destination", "Index", "FF Time"]

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                table.add_row([od[0], od[1], idx, "%.3f"%time])
            table.add_row(["----", "----", "----", "----"])

        print("------ Free flow travel times ------")
        print(table)