import pandas as pd

from prettytable import PrettyTable

from keychain import Keychain as kc
from services import Simulator
from utilities import make_dir


class TrafficEnvironment:

    def __init__(self, environment_params, simulation_params):
        self.simulator = Simulator(simulation_params)
        self.action_space_size = environment_params[kc.ACTION_SPACE_SIZE]

        self.joint_action_cols = [kc.AGENT_ID, kc.AGENT_KIND, kc.ACTION, kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME]
        self.joint_action = pd.DataFrame(columns = self.joint_action_cols)

        print("[SUCCESS] Environment initiated!")


    def start(self):
        self.simulator.start()

    def stop(self):
        self.simulator.stop()

    def reset(self):
        self.joint_action = pd.DataFrame(columns = self.joint_action_cols)
        self.simulator.reset()


    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self._print_free_flow_times(free_flow_times)
        self._save_free_flow_times_csv(free_flow_times)
        return free_flow_times
    

    def get_observation(self):
        return self.joint_action


    def register_action(self, agent, action):
        action_data = [agent.id, agent.kind, action, agent.origin, agent.destination, agent.start_time]
        self.joint_action.loc[len(self.joint_action.index)] = {key : value for key, value in zip(self.joint_action_cols, action_data)}


    def step(self):
        sumo_df = self.simulator.simulate_episode(self.joint_action)
        sumo_df = sumo_df[[kc.AGENT_ID, kc.TRAVEL_TIME]]
        joint_observation = sumo_df.merge(self.joint_action, on=kc.AGENT_ID)
        info = {kc.LAST_SIM_DURATION: self.get_last_sim_duration()}
        return joint_observation, info


    def get_last_sim_duration(self):
        return self.simulator.last_simulation_duration


    def _print_free_flow_times(self, free_flow_times):
        table = PrettyTable()
        table.field_names = ["Origin", "Destination", "Index", "FF Time"]

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                table.add_row([od[0], od[1], idx, "%.3f"%time])
            table.add_row(["----", "----", "----", "----"])

        print("------ Free flow travel times ------")
        print(table)


    def _save_free_flow_times_csv(self, free_flow_times):
        cols = [kc.ORIGINS, kc.DESTINATIONS, kc.PATH_INDEX, kc.FREE_FLOW_TIME]
        free_flow_pd = pd.DataFrame(columns=cols)

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                free_flow_pd.loc[len(free_flow_pd.index)] = [od[0], od[1], idx, time]
        save_to = make_dir(kc.RECORDS_FOLDER, kc.FREE_FLOW_TIMES_CSV_FILE_NAME)
        free_flow_pd.to_csv(save_to, index = False)
        print(f"[SUCCESS] Free-flow travel times saved to: {save_to}")