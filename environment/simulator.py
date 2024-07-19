import os
import logging
import pandas as pd
import random
import traci

from keychain import Keychain as kc
from utilities import confirm_env_variable

import time
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class SumoSimulator():

    def __init__(self, params):
        self.sumo_config_path = kc.SUMO_CONFIG_PATH
        self.paths_csv_path = kc.PATHS_CSV_SAVE_PATH
        self.routes_xml_path = kc.ROUTES_XML_SAVE_PATH

        self.sumo_type = params[kc.SUMO_TYPE]
        self.env_var = params[kc.ENV_VAR]
        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS]
        self.seed = params[kc.SEED] 

        ## Detectors
        self.detectors_name = list(pd.read_csv(kc.PATHS_CSV_SAVE_DETECTORS).name) ###FIX THIS
        self.det_dict = {name: [] for name in self.detectors_name}

        self.sumo_id = f"{random.randint(0, 1000)}"
        self.sumo_connection = None

        self._check_paths_ready()
        confirm_env_variable(self.env_var, append="tools")
        
        self.timestep = 0
        self.route_id_cache = dict()

        logging.info("[SUCCESS] Simulator is ready to simulate!")

    #####################

    ##### CONFIG CHECK #####

    def _check_paths_ready(self):
        if os.path.isfile(self.paths_csv_path) and os.path.isfile(self.routes_xml_path):
            logging.info("[CONFIRMED] Paths file is ready.")
        else:
            raise FileNotFoundError("Paths file is not ready. Please generate paths first.")
        
    #####################

    ##### SUMO CONTROL #####

    def start(self):
        sumo_cmd = [self.sumo_type,"--seed", self.seed, "-c", self.sumo_config_path] 
        traci.start(sumo_cmd, label=self.sumo_id)
        self.sumo_connection = traci.getConnection(self.sumo_id)

    def stop(self):
        self.sumo_connection.close()

    def reset(self):##make empty the det_dict
        self.sumo_connection.load(["--seed", self.seed,'-c', self.sumo_config_path])
        self.timestep = 0

    #####################

    ##### SIMULATION #####
    
    def add_vehice(self, act_dict: dict):
        route_id = self.route_id_cache.setdefault((act_dict[kc.AGENT_ORIGIN], act_dict[kc.AGENT_DESTINATION], act_dict[kc.ACTION]), \
                f'{act_dict[kc.AGENT_ORIGIN]}_{act_dict[kc.AGENT_DESTINATION]}_{act_dict[kc.ACTION]}')
        self.sumo_connection.vehicle.add(vehID=str(act_dict[kc.AGENT_ID]), routeID=route_id, depart=str(act_dict[kc.AGENT_START_TIME]))
    
    def step(self):
        arrivals = self.sumo_connection.simulation.getArrivedIDList()
        self.sumo_connection.simulationStep()
        self.timestep += 1

        #### Function
        """for id, name in enumerate(self.detectors_name):
            
            link = self.sumo_connection.inductionloop.getIntervalVehicleNumber(f"{name}_det")
            self.det_dict[name] = ((link / self.timestep) * 3600) # 1hour"""
        self.det_dict = []
        
        return self.timestep, arrivals, self.det_dict
    
    #####################
    