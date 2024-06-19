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
    def step(self):
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
        
        self.timestep = 0
        self.vehicles_in_network = dict()

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
        self.timestep = 0
        self.vehicles_in_network = dict()

    #####################

    ##### SIMULATION #####
    
    def step(self, actions):
        
        if not actions.empty:
            actions[kc.SUMO_ACTION] = actions.apply(self.sumonize_action, axis=1)
            actions.apply(self.add_to_sim, axis=1)
            for _, row in actions.iterrows():
                self.vehicles_in_network[row[kc.AGENT_ID]] = row[kc.AGENT_START_TIME]
        
        travel_times = dict()
        for veh_id in self.sumo_connection.simulation.getArrivedIDList():
            travel_times[int(veh_id)] = (self.timestep - self.vehicles_in_network[int(veh_id)]) / 60.0
            del self.vehicles_in_network[int(veh_id)]
        travel_time_df = pd.DataFrame({kc.AGENT_ID: list(travel_times.keys()), kc.TRAVEL_TIME: list(travel_times.values())})
        
        self.sumo_connection.simulationStep()
        self.timestep += 1
        return self.timestep, travel_time_df
        
    
    #####################
    
    def get_free_flow_times(self):
        paths_df = pd.read_csv(self.paths_csv_path)
        origins = paths_df[kc.ORIGIN].unique()
        destinations = paths_df[kc.DESTINATION].unique()
        ff_dict = {(o, d): list() for o in origins for d in destinations}
        for _, row in paths_df.iterrows():
            ff_dict[(row[kc.ORIGIN], row[kc.DESTINATION])].append(row[kc.FREE_FLOW_TIME])
        return ff_dict