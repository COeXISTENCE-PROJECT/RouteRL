import os
import logging
import random

import janux as jx
import pandas as pd
import traci

from ..keychain import Keychain as kc
from ..utilities import confirm_env_variable

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class SumoSimulator():
    """ 

    A class responsible for managing the communication between our learning agents and the SUMO traffic simulator.
    SUMO provides the traffic environment where vehicles travel between designated origins and destinations,
    and it returns the corresponding travel times for these vehicles.

    """

    def __init__(self, params:dict, path_gen_params: dict|None = None) -> None:
        self.sumo_config_path = params[kc.SUMO_CONFIG_PATH]
        self.routes_xml_path = params[kc.ROUTE_FILE_PATH]
        self.paths_csv_path = params[kc.PATHS_CSV_SAVE_PATH]
        self.sumo_fcd = params[kc.SUMO_FCD]

        self.sumo_type = params[kc.SUMO_TYPE]
        self.env_var = params[kc.ENV_VAR]
        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS]
        self.seed = str(params[kc.SEED])
        self.detector_save_path = params[kc.PATHS_CSV_SAVE_DETECTORS]

        ## Detectors
        self.detectors_name = list(pd.read_csv(self.detector_save_path).name) ###FIX THIS

        self.sumo_id = f"{random.randint(0, 1000)}"
        self.sumo_connection = None

        if path_gen_params is not None:
            self._get_paths(params, path_gen_params)
            logging.info("[SUCCESS] Path generation completed.")

            
        self._check_paths_ready()
        confirm_env_variable(self.env_var, append="tools")
        
        self.timestep = 0
        self.route_id_cache = dict()

        logging.info("[SUCCESS] Simulator is ready to simulate!")

    #####################

    ##### CONFIG CHECK #####

    def _check_paths_ready(self) -> None:
        """
        Checks if the required paths file for the simulation exists.

        """
        if os.path.isfile(self.routes_xml_path):
            logging.info("[CONFIRMED] Paths file is ready.")
        else:
            raise FileNotFoundError(
                "Paths file is not ready. Please generate paths first.\n"
                "To do this, define arguments in params.json and pass it to the environment under path_generation_parameters"
            )
            
    def _get_paths(self, params: dict, path_gen_params: dict) -> pd.DataFrame:
        # Build the network
        network = jx.build_digraph(params[kc.CONNECTION_FILE_PATH], params[kc.EDGE_FILE_PATH], self.routes_xml_path)
        
        # Generate paths
        origins = path_gen_params[kc.ORIGINS]
        destinations = path_gen_params[kc.DESTINATIONS]
        path_gen_kwargs = {
            "number_of_paths": path_gen_params[kc.NUMBER_OF_PATHS],
            "random_seed": int(self.seed),
            "num_samples": path_gen_params[kc.NUM_SAMPLES],
            "beta": path_gen_params[kc.BETA],
            "weight": path_gen_params[kc.WEIGHT],
            "verbose": False
        }
        routes = jx.basic_generator(network, origins, destinations, as_df=True, calc_free_flow=True, **path_gen_kwargs)
        self._save_paths_to_disk(routes, origins, destinations, params[kc.ROUTE_SAVE_FILE_PATH])
        
        # Save paths visualizations
        path_visuals_path = params[kc.FIGURES_SAVE_PATH]
        os.makedirs(path_visuals_path, exist_ok=True)
        # Visualize paths and save figures
        for origin_idx, origin in enumerate(origins):
            for dest_idx, destination in enumerate(destinations):
                # Filter routes for the current origin-destination pair
                routes_to_show = routes[(routes["origins"] == origin_idx) & (routes["destinations"] == dest_idx)]['path']
                routes_to_show = [route.split(" ") for route in routes_to_show]
                # Specify the save path and title for the figure
                fig_save_path = os.path.join(path_visuals_path, f"{origin_idx}_{dest_idx}.png")
                title=f"Origin: {origin_idx} ({origin}), Destination: {dest_idx} ({destination})"
                # Show the routes
                jx.show_multi_routes(params[kc.NOD_FILE_PATH], params[kc.EDGE_FILE_PATH],
                                     routes_to_show, origin, destination, 
                                     show=False, save_file_path=fig_save_path, title=title)
        
            
    def _save_paths_to_disk(self, routes_df: pd.DataFrame, origins: list, destinations: list, save_path: str) -> None:
        origins = {node_name: idx for idx, node_name in enumerate(origins)}
        destinations = {node_name: idx for idx, node_name in enumerate(destinations)}
        
        # Format routes dataframe
        routes_df["origins"] = routes_df["origins"].apply(lambda x: origins[x])
        routes_df["destinations"] = routes_df["destinations"].apply(lambda x: destinations[x])
        routes_df["path"] = routes_df["path"].apply(lambda x: x.replace(",", " "))
        routes_df.sort_values(by=["origins", "destinations"], inplace=True)
        
        # Save paths to csv
        routes_df.to_csv(self.paths_csv_path, index=False)
        
        # Convert routes dataframe to a dictionary with od indices
        paths_dict = dict()
        for origin_idx in origins.values():
            for destination_idx in destinations.values():
                rows = routes_df[(routes_df["origins"] == origin_idx) & (routes_df["destinations"] == destination_idx)]
                paths = list(rows["path"].values)
                paths_dict[(origin_idx, destination_idx)] = paths
                
        # Save paths to xml
        with open(save_path, "w") as rou:
            print("""<routes>""", file=rou)
            # TODO: Following two lines are hardcoded. Change them to be dynamic.
            print("<vType id=\"Human\" color=\"red\" guiShape=\"passenger/sedan\"/>", file=rou)
            print("<vType id=\"AV\" color=\"yellow\"/>", file=rou)
            for origin_idx in origins.values():
                for destination_idx in destinations.values():
                    paths = routes_df[(routes_df["origins"] == origin_idx) & (routes_df["destinations"] == destination_idx)]["path"].values
                    for idx, path in enumerate(paths):
                        print(f'<route id="{origin_idx}_{destination_idx}_{idx}" edges="', file=rou)
                        print(path, file=rou)
                        print('" />',file=rou)
            print("</routes>", file=rou)

    #####################

    ##### SUMO CONTROL #####

    def start(self) -> None:
        """
        Starts the SUMO simulation with the specified configuration.
        """

        sumo_cmd = [self.sumo_type,"--seed", self.seed, "--fcd-output", self.sumo_fcd, "-c", self.sumo_config_path] 
        traci.start(sumo_cmd, label=self.sumo_id)
        self.sumo_connection = traci.getConnection(self.sumo_id)

    def stop(self) -> None:
        """
        Stops and closes the SUMO simulation.
        """
        self.sumo_connection.close()

    def reset(self) -> None:
        """
        Resets the SUMO simulation to its initial state.
        Reads detector data.
        """

        det_dict = {name: None for name in self.detectors_name}
        for det_name in self.detectors_name:
            det_dict[det_name]  = self.sumo_connection.inductionloop.getIntervalVehicleNumber(f"{det_name}_det")

        self.sumo_connection.load(["--seed", self.seed, "--fcd-output", self.sumo_fcd, '-c', self.sumo_config_path])

        self.timestep = 0
        return det_dict

    #####################

    ##### SIMULATION #####
    
    def add_vehice(self, act_dict: dict) -> None:
        """
        Adds a vehicle to the SUMO simulation environment with the specified route and parameters.

        Parameters:
        - act_dict (dict): A dictionary containing key vehicle attributes.

        """

        route_id = self.route_id_cache.setdefault((act_dict[kc.AGENT_ORIGIN], act_dict[kc.AGENT_DESTINATION], act_dict[kc.ACTION]), \
                f'{act_dict[kc.AGENT_ORIGIN]}_{act_dict[kc.AGENT_DESTINATION]}_{act_dict[kc.ACTION]}')
        kind = act_dict[kc.AGENT_KIND]
        self.sumo_connection.vehicle.add(vehID=str(act_dict[kc.AGENT_ID]), routeID=route_id, depart=str(act_dict[kc.AGENT_START_TIME]), typeID=kind)
    

    def step(self) -> tuple:
        """
        Advances the SUMO simulation by one timestep and retrieves information about vehicle arrivals and detector data.

        Returns:
            tuple: A tuple containing:
                self.timestep (int): The current simulation timestep.
                arrivals (list): List of vehicle IDs that arrived at their destinations during the current timestep.   
        """
   
        arrivals = self.sumo_connection.simulation.getArrivedIDList()
        self.sumo_connection.simulationStep()
        self.timestep += 1
        
        return self.timestep, arrivals
    
    #####################
    