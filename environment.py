import pandas as pd

from prettytable import PrettyTable

from keychain import Keychain as kc
from services import Simulator
from utilities import create_agent_objects
from utilities import make_dir


class TrafficEnvironment:

    def __init__(self, environment_params, simulation_params, agent_params, render_mode=None):
        self.simulator = Simulator(simulation_params)
        self.agents = create_agent_objects(agent_params, self.calculate_free_flow_times())
        self.action_space_size = environment_params[kc.ACTION_SPACE_SIZE]

        self.possible_agents = [agent.id for agent in self.agents] 

        self.joint_action_cols = [kc.AGENT_ID, kc.AGENT_KIND, kc.ACTION, kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME]
        self.joint_action = pd.DataFrame(columns = self.joint_action_cols)

        self.options_last_picked = dict()

        print("[SUCCESS] Environment initiated!")


    def start(self):
        self.simulator.start_sumo()

    def stop(self):
        self.simulator.stop_sumo()

    def reset(self):
        self.joint_action = pd.DataFrame(columns = self.joint_action_cols)
        self.options_last_picked = dict()
        self.simulator.reset_sumo()


    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self.print_free_flow_times(free_flow_times)
        self.save_free_flow_times_csv(free_flow_times)
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
        self.update_options_last_picked(agent, action)

    
    def update_options_last_picked(self, agent, action):
        action_key = f"{agent.origin}_{agent.destination}_{action}"
        self.options_last_picked[action_key] = max(self.options_last_picked.get(action_key, -1), agent.start_time)


    def step(self):        

        sumo_df = self.simulator.run_simulation_iteration(self.joint_action)
        joint_observation = self.prepare_observations(sumo_df, self.joint_action)

        info = {kc.LAST_SIM_DURATION: self.get_last_sim_duration()}

        return joint_observation, info


    def prepare_observations(self, sumo_df, joint_action):
        # Calculate reward from cost (skipped)
        observation_df = sumo_df.merge(joint_action, on=kc.AGENT_ID)

        # Machines rewards (to minimize) = 0.4 x own_time + 0.6 x machines_mean_time
        machines_mean_travel_times = observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE, kc.TRAVEL_TIME].mean()
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE, kc.TRAVEL_TIME] *= 0.4
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE, kc.TRAVEL_TIME] += (machines_mean_travel_times * 0.6)

        # Malicious machines rewards (to maximize) = 0.3 x -machines2_mean_travel_time + 0.7 x machines_mean_time
        machines_mean_travel_times = observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE, kc.TRAVEL_TIME].mean()
        machines2_mean_travel_times = observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.TRAVEL_TIME].mean()
        #observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.TRAVEL_TIME] *= -0.2
        #observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.TRAVEL_TIME] += (-1 * (machines2_mean_travel_times * 0.3))
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.TRAVEL_TIME] = (-1 * (machines2_mean_travel_times * 0.3))
        observation_df.loc[observation_df[kc.AGENT_KIND] == kc.TYPE_MACHINE_2, kc.TRAVEL_TIME] += (machines_mean_travel_times * 0.7)

        return observation_df[[kc.AGENT_ID, kc.TRAVEL_TIME]]


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


    def save_free_flow_times_csv(self, free_flow_times):
        cols = [kc.ORIGINS, kc.DESTINATIONS, kc.PATH_INDEX, kc.FREE_FLOW_TIME]
        free_flow_pd = pd.DataFrame(columns=cols)

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                free_flow_pd.loc[len(free_flow_pd.index)] = [od[0], od[1], idx, time]
        save_to = make_dir(kc.RECORDS_FOLDER, kc.FREE_FLOW_TIMES_CSV_FILE_NAME)
        free_flow_pd.to_csv(save_to, index = False)
        print(f"[SUCCESS] Free-flow travel times calculated and saved to: {save_to}")