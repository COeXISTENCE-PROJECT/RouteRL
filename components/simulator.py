import os
import pandas as pd
import random
import traci

from abc import ABC, abstractmethod

from keychain import Keychain as kc
from utilities import confirm_env_variable

class BaseSimulator(ABC):

    """
    The interface between the simulation software and the environment
    """

    def __init__(self,):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def simulate_episode(self):
        pass



class SumoSimulator(BaseSimulator):

    def __init__(self, params):
        super().__init__()

        self.sumo_config_path = kc.SUMO_CONFIG_PATH
        self.paths_csv_path = kc.PATHS_CSV_SAVE_PATH
        self.routes_xml_path = kc.ROUTES_XML_SAVE_PATH

        self.sumo_type = params[kc.SUMO_TYPE]
        self.env_var = params[kc.ENV_VAR]
        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS]

        self.sumo_id = f"{random.randint(0, 1000)}"
        self.sumo_connection = None
        
        self.sumonize_action = lambda row: f'{row[kc.AGENT_ORIGIN]}_{row[kc.AGENT_DESTINATION]}_{row[kc.ACTION]}'
        self.add_to_sim = lambda row: self.sumo_connection.vehicle.add(vehID=f"{row[kc.AGENT_ID]}", routeID=row[kc.SUMO_ACTION], depart=f"{row[kc.AGENT_START_TIME]}")

        self._check_paths_ready()
        confirm_env_variable(self.env_var, append="tools")

        print("[SUCCESS] Simulator is ready to simulate!")

    #####################

    ##### CONFIG CHECK #####

    def _check_paths_ready(self):
        if os.path.isfile(self.paths_csv_path) and os.path.isfile(self.routes_xml_path):
            print("[CONFIRMED] Paths file is ready.")
        else:
            raise FileNotFoundError("Paths file is not ready. Please generate paths first.")
        
    #####################

    ##### SUMO CONTROL #####

    def start(self):
        sumo_cmd = [self.sumo_type, "-c", self.sumo_config_path]
        traci.start(sumo_cmd, label=self.sumo_id)
        self.sumo_connection = traci.getConnection(self.sumo_id)

    def stop(self):
        self.sumo_connection.close()

    def reset(self):
        self.sumo_connection.load(['-c', self.sumo_config_path])

    #####################

    ##### SIMULATION #####

    def simulate_episode(self, joint_action):
        num_vehicles = len(joint_action)
        # Construct SUMO actions
        joint_action[kc.SUMO_ACTION] = joint_action.apply(self.sumonize_action, axis=1)
        # Add all vehicles to the simulation at once
        joint_action.apply(self.add_to_sim, axis=1)
        # Collect arrivals
        arrivals = {kc.AGENT_ID : list(), kc.ARRIVAL_TIME : list()}
        timestep = 0
        while len(arrivals[kc.AGENT_ID]) < num_vehicles:
            self.sumo_connection.simulationStep()
            timestep += 1
            for veh_id in self.sumo_connection.simulation.getArrivedIDList():
                arrivals[kc.AGENT_ID].append(int(veh_id))
                arrivals[kc.ARRIVAL_TIME].append(timestep)
        # Calculate travel times df
        return self._prepare_travel_times_df(arrivals, joint_action)
        

    def _prepare_travel_times_df(self, arrivals, joint_action):
        # Initiate the travel_time_df
        travel_times_df = pd.DataFrame(arrivals)
        # Retrieve the start times of the agents from the joint_action dataframe
        start_times_df = joint_action[[kc.AGENT_ID, kc.AGENT_START_TIME]]
        # Merge the travel_time_df with the start_times_df for travel time calculation
        travel_times_df = pd.merge(left=start_times_df, right=travel_times_df, on=kc.AGENT_ID, how='left')
        # Calculate travel time
        calculate_travel_duration = lambda row: ((row[kc.ARRIVAL_TIME] - row[kc.AGENT_START_TIME]) / 60.0)
        travel_times_df[kc.TRAVEL_TIME] = travel_times_df.apply(calculate_travel_duration, axis=1)
        # Retain only the necessary columns
        return travel_times_df[[kc.AGENT_ID, kc.TRAVEL_TIME]]
    
    #####################
    
    def get_free_flow_times(self):
        paths_df = pd.read_csv(self.paths_csv_path)
        origins = paths_df[kc.ORIGIN].unique()
        destinations = paths_df[kc.DESTINATION].unique()
        ff_dict = {(o, d): list() for o in origins for d in destinations}
        for _, row in paths_df.iterrows():
            ff_dict[(row[kc.ORIGIN], row[kc.DESTINATION])].append(row[kc.FREE_FLOW_TIME])
        return ff_dict