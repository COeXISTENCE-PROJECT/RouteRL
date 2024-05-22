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

        self.options_last_picked = dict()

        print("[SUCCESS] Environment initiated!")


    def start(self):
        self.simulator.start()

    def stop(self):
        self.simulator.stop()

    def reset(self):
        self.joint_action = pd.DataFrame(columns = self.joint_action_cols)
        self.options_last_picked = dict()
        self.simulator.reset()


    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self._print_free_flow_times(free_flow_times)
        self._save_free_flow_times_csv(free_flow_times)
        return free_flow_times
    

    def get_observation(self, agent_type, origin, destination):
        if agent_type == kc.TYPE_HUMAN:
            return None
        if agent_type == kc.TYPE_MACHINE:
            observation = list()
            for possible_action in range(self.action_space_size):
                action_key = f"{origin}_{destination}_{possible_action}"
                obs = self.options_last_picked.get(action_key, -1)
                observation.append(obs)
            return observation
        if agent_type == kc.TYPE_MACHINE_2:
            return self.joint_action.loc[(self.joint_action[kc.AGENT_KIND] == kc.TYPE_MACHINE) & (self.joint_action[kc.AGENT_ORIGIN] == origin) & (self.joint_action[kc.AGENT_DESTINATION] == destination)]
    

    def register_action(self, agent, action):
        action_data = [agent.id, agent.kind, action, agent.origin, agent.destination, agent.start_time]
        self.joint_action.loc[len(self.joint_action.index)] = {key : value for key, value in zip(self.joint_action_cols, action_data)}
        self._update_options_last_picked(agent, action)

    
    def _update_options_last_picked(self, agent, action):
        action_key = f"{agent.origin}_{agent.destination}_{action}"
        self.options_last_picked[action_key] = max(self.options_last_picked.get(action_key, -1), agent.start_time)


    def step(self):   
        sumo_df = self.simulator.simulate_episode(self.joint_action)
        joint_observation = self._prepare_rewards(sumo_df, self.joint_action)
        info = {kc.LAST_SIM_DURATION: self.get_last_sim_duration()}
        return joint_observation, info


    def _prepare_rewards(self, sumo_df, joint_action):
        # Calculate reward from cost (skipped)
        observation_df = sumo_df.merge(joint_action, on=kc.AGENT_ID)
        observation_df[kc.REWARD] = observation_df[kc.TRAVEL_TIME]

        # It will remain the same for humans
        # Machines rewards (to minimize) = 0.5 x own_time + 0.5 x machines_mean_time
        machines_mean_travel_times = observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE, kc.TRAVEL_TIME].mean()
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE, kc.REWARD] *= 0.5
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE, kc.REWARD] += (machines_mean_travel_times * 0.5)

        # Disruptive machines rewards (to minimize) = 0.3 x own_time + 0.3 x machines2_mean_travel_time - 0.4 x humans_mean_time
        machines2_mean_travel_times = observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.TRAVEL_TIME].mean()
        humans_mean_travel_times = observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_HUMAN, kc.TRAVEL_TIME].mean()
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.REWARD] *= 0.3
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.REWARD] += (0.3 * machines2_mean_travel_times)
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.REWARD] -= (0.4 * humans_mean_travel_times)

        return observation_df[[kc.AGENT_ID, kc.TRAVEL_TIME, kc.REWARD]]


    def get_last_sim_duration(self):
        return self.simulator.get_last_sim_duration()


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