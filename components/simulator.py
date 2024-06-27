import os
import pandas as pd
import random
import traci

from keychain import Keychain as kc
from utilities import confirm_env_variable

import time
import numpy as np
class SumoSimulator():

    def __init__(self, params):
        self.sumo_config_path = kc.SUMO_CONFIG_PATH
        self.paths_csv_path = kc.PATHS_CSV_SAVE_PATH
        self.routes_xml_path = kc.ROUTES_XML_SAVE_PATH

        self.sumo_type = params[kc.SUMO_TYPE]
        self.env_var = params[kc.ENV_VAR]
        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS]

        self.sumo_id = f"{random.randint(0, 1000)}"
        self.sumo_connection = None

        self._check_paths_ready()
        confirm_env_variable(self.env_var, append="tools")
        
        self.timestep = 0

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

    #####################

    ##### SIMULATION #####
    
    def step(self, actions):
        for vehicle_id, data in actions.items():
            route_id = f'{data[kc.AGENT_ORIGIN]}_{data[kc.AGENT_DESTINATION]}_{data[kc.ACTION]}'
            self.sumo_connection.vehicle.add(vehID=str(vehicle_id), routeID=route_id, depart=str(data[kc.AGENT_START_TIME]))
        arrivals = list(self.sumo_connection.simulation.getArrivedIDList())
        self.sumo_connection.simulationStep()
        self.timestep += 1
        return self.timestep, arrivals
    
    #####################
    
    def get_free_flow_times(self):
        paths_df = pd.read_csv(self.paths_csv_path)
        origins = paths_df[kc.ORIGIN].unique()
        destinations = paths_df[kc.DESTINATION].unique()
        ff_dict = {(o, d): list() for o in origins for d in destinations}
        for _, row in paths_df.iterrows():
            ff_dict[(row[kc.ORIGIN], row[kc.DESTINATION])].append(row[kc.FREE_FLOW_TIME])
        return ff_dict