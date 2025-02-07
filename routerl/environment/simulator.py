"""
This file, simulator.py, defines the SumoSimulator class, 
which manages the interaction between driver agents and the SUMO traffic simulator 
by handling simulation setup, path generation, vehicle management, and simulation control.
"""
import os
import janux as jx
import logging
import random
import pandas as pd
import traci

from routerl.keychain import Keychain as kc
from routerl.utilities import confirm_env_variable

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class SumoSimulator():
    """Sumo simulator class

    A class responsible for managing the communication between our learning agents and the SUMO traffic simulator.
    SUMO provides the traffic environment where vehicles travel between designated origins and destinations,
    and it returns the corresponding travel times for these vehicles.

    Args:
        params (dict):
            Dictionary of parameters for the SUMO environment. Specified `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        path_gen_params (dict):
            Dictionary of parameters for the SUMO environment. specified `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        seed (int): 
            Random seed for reproducibility.
        
    Attributes:
        network_name: Network name.
        simulation_length: Simulation length.
        sumo_id: SUMO connection id.
        sumo_connection: Traci-SUMO connection object.
        timestep: Time step being simulated within the day.
    """

    def __init__(self, params: dict, path_gen_params: dict, seed: int = 23423):
        self.network_name        = params[kc.NETWORK_NAME]
        self.sumo_type           = params[kc.SUMO_TYPE]
        self.number_of_paths     = params[kc.NUMBER_OF_PATHS]
        self.simulation_length   = params[kc.SIMULATION_TIMESTEPS]

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        self.network_folder      = os.path.join(curr_dir,
                                                kc.NETWORK_FOLDER).replace("$net$", self.network_name)
        self.sumo_config_path    = os.path.join(curr_dir,
                                                kc.SUMO_CONFIG_PATH).replace("$net$", self.network_name)
        self.routes_xml_path     = os.path.join(curr_dir,
                                                kc.ROU_FILE_PATH).replace("$net$", self.network_name)
        self.sumo_fcd            = os.path.join(curr_dir,
                                                kc.SUMO_FCD).replace("$net$", self.network_name)
        self.detector_save_path  = os.path.join(curr_dir,
                                                kc.DETECTORS_CSV_PATH).replace("$net$", self.network_name)
        self.conn_file_path      = os.path.join(curr_dir,
                                                kc.CONNECTION_FILE_PATH).replace("$net$", self.network_name)
        self.edge_file_path      = os.path.join(curr_dir,
                                                kc.EDGE_FILE_PATH).replace("$net$", self.network_name)
        self.nod_file_path       = os.path.join(curr_dir,
                                                kc.NOD_FILE_PATH).replace("$net$", self.network_name)
        self.rou_xml_save_path   = os.path.join(curr_dir,
                                                kc.ROUTE_XML_PATH).replace("$net$", self.network_name)
        self.det_xml_save_path   = os.path.join(curr_dir,
                                                kc.DETECTORS_XML_PATH).replace("$net$", self.network_name)
        self.default_od_path     = os.path.join(curr_dir,
                                                kc.DEFAULT_ODS_PATH)
        
        self.paths_csv_file_path = os.path.join(params[kc.RECORDS_FOLDER], kc.PATHS_CSV_FILE_NAME)

        random.seed(seed)

        self.seed = seed
        self.sumo_id = f"{random.randint(0, 1000)}"
        self.sumo_connection = None

        confirm_env_variable(kc.ENV_VAR, append="tools")

        if path_gen_params is not None:
            self._get_paths(params, path_gen_params)
            logging.info("[SUCCESS] Path generation completed.")
        self._check_paths_ready()
        self.detectors_name = self._get_detectors()
        self.timestep = 0
        self.route_id_cache = dict()

        logging.info("[SUCCESS] Simulator is ready to simulate!")

    ################################
    ######## CONFIG CHECKS #########
    ################################

    def _check_paths_ready(self) -> None:

        if os.path.isfile(self.rou_xml_save_path):
            logging.info("[CONFIRMED] Paths file is ready.")
        else:
            raise FileNotFoundError(
                "Paths file is not ready. Please generate paths first.\n"
                "To do this, define arguments in params.json and pass it "
                "to the environment under path_generation_parameters"
            )
            
    def _get_paths(self, params: dict, path_gen_params: dict) -> None:

        # Build the network
        network = jx.build_digraph(self.conn_file_path, self.edge_file_path, self.routes_xml_path)
        
        # Get origins and destinations
        origins = path_gen_params[kc.ORIGINS]
        destinations = path_gen_params[kc.DESTINATIONS]
        
        # Generate paths
        path_gen_kwargs = {
            "number_of_paths": path_gen_params[kc.NUMBER_OF_PATHS],
            "random_seed": self.seed,
            "num_samples": path_gen_params[kc.NUM_SAMPLES],
            "beta": path_gen_params[kc.BETA],
            "weight": path_gen_params[kc.WEIGHT],
            "verbose": False
        }
        routes = jx.basic_generator(network, origins, destinations, as_df=True, calc_free_flow=True, **path_gen_kwargs)
        self._save_paths_to_disc(routes, origins, destinations)
        
        # Save paths visualizations
        path_visuals_path = params[kc.PLOTS_FOLDER]
        os.makedirs(path_visuals_path, exist_ok=True)
        # Visualize paths and save figures
        for origin_idx, origin in enumerate(origins):
            for dest_idx, destination in enumerate(destinations):
                # Filter routes for the current origin-destination pair
                routes_to_show = (routes[(routes["origins"] == origin_idx)
                                         & (routes["destinations"] == dest_idx)]['path'])
                routes_to_show = [route.split(" ") for route in routes_to_show]
                # Specify the save path and title for the figure
                fig_save_path = os.path.join(path_visuals_path, f"{origin_idx}_{dest_idx}.png")
                title=f"Origin: {origin_idx} ({origin}), Destination: {dest_idx} ({destination})"
                # Show the routes
                jx.show_multi_routes(self.nod_file_path, self.edge_file_path,
                                    routes_to_show, origin, destination, 
                                    show=False, save_file_path=fig_save_path, title=title)

    def _save_paths_to_disc(self, routes_df: pd.DataFrame, origins: list, destinations: list) -> None:

        origins = {node_name: idx for idx, node_name in enumerate(origins)}
        destinations = {node_name: idx for idx, node_name in enumerate(destinations)}
        
        # Format routes dataframe
        routes_df["origins"] = routes_df["origins"].apply(lambda x: origins[x])
        routes_df["destinations"] = routes_df["destinations"].apply(lambda x: destinations[x])
        routes_df["path"] = routes_df["path"].apply(lambda x: x.replace(",", " "))
        routes_df.sort_values(by=["origins", "destinations"], inplace=True)

        # Save paths to csv
        routes_df.to_csv(self.paths_csv_file_path, index=False)

        # Convert routes dataframe to a dictionary with od indices
        paths_dict = dict()
        for origin_idx in origins.values():
            for destination_idx in destinations.values():
                rows = routes_df[(routes_df["origins"] == origin_idx) & (routes_df["destinations"] == destination_idx)]
                paths = rows["path"].to_list()
                paths_dict[(origin_idx, destination_idx)] = paths

        # Save paths to xml
        with open(self.rou_xml_save_path, "w") as rou:
            print("""<routes>""", file=rou)
            # TODO: Following two lines are hardcoded. Change them to be dynamic.
            print("<vType id=\"Human\" color=\"red\" guiShape=\"passenger/sedan\"/>", file=rou)
            print("<vType id=\"AV\" color=\"yellow\"/>", file=rou)
            for origin_idx in origins.values():
                for destination_idx in destinations.values():
                    paths = (routes_df[(routes_df["origins"] == origin_idx)
                                       & (routes_df["destinations"] == destination_idx)]["path"].values)
                    for idx, path in enumerate(paths):
                        print(f'<route id="{origin_idx}_{destination_idx}_{idx}" edges="', file=rou)
                        print(path, file=rou)
                        print('" />',file=rou)
            print("</routes>", file=rou)

    def _get_detectors(self):

        paths_df = pd.read_csv(self.paths_csv_file_path)
        paths_list = [path.split(" ") for path in paths_df["path"].values]
        detectors_name = sorted(list(set([node for path in paths_list for node in path])))
        
        detectors_df = pd.DataFrame({"name": detectors_name})
        detectors_df.to_csv(self.detector_save_path, index=False)
        
        with open(self.det_xml_save_path, "w") as det:
            print("""<additional>""", file=det)
            for det_id in detectors_name:
                print(f"<inductionLoop id=\"{det_id}_det\" lane=\"{det_id}_0\" pos=\"-5\" file=\"NUL\" friendlyPos=\"True\"/>",
                      file=det)
            print("</additional>", file=det)
            
        return detectors_name

    ################################
    ######## SUMO CONTROL ##########
    ################################

    def start(self) -> None:
        """Starts the SUMO simulation with the specified configuration.

        Returns:
            None
        """

        sumo_cmd = [self.sumo_type,"--seed",
                    str(self.seed),
                    "--fcd-output",
                    self.sumo_fcd,
                    "-c", self.sumo_config_path]
        traci.start(sumo_cmd, label=self.sumo_id)
        self.sumo_connection = traci.getConnection(self.sumo_id)

    def stop(self) -> None:
        """Stops and closes the SUMO simulation.

        Returns:
            None
        """

        self.sumo_connection.close()

    def reset(self) -> dict:
        """ Resets the SUMO simulation to its initial state.

        Reads detector data.

        Returns:
            det_dict (dict[str, float]): dictionary with detector data.
        """

        det_dict = {name: None for name in self.detectors_name}
        for det_name in self.detectors_name:
            det_dict[det_name]  = self.sumo_connection.inductionloop.getIntervalVehicleNumber(f"{det_name}_det")

        self.sumo_connection.load(["--seed",
                                   str(self.seed),
                                   "--fcd-output",
                                   self.sumo_fcd,
                                   '-c',
                                   self.sumo_config_path])

        self.timestep = 0
        return det_dict

    ################################
    ######### SIMULATION ###########
    ################################

    def add_vehicle(self, act_dict: dict) -> None:
        """Adds a vehicle to the SUMO simulation environment with the specified route and parameters.

        Args:
            act_dict (dict): A dictionary containing key vehicle attributes.
        Returns:
            None
        """

        route_id = (
            self.route_id_cache.setdefault((
                act_dict[kc.AGENT_ORIGIN],
                act_dict[kc.AGENT_DESTINATION],
                act_dict[kc.ACTION]),
                f'{act_dict[kc.AGENT_ORIGIN]}_{act_dict[kc.AGENT_DESTINATION]}_{act_dict[kc.ACTION]}'))
        kind = act_dict[kc.AGENT_KIND]
        self.sumo_connection.vehicle.add(vehID=str(act_dict[kc.AGENT_ID]),
                                         routeID=route_id,
                                         depart=str(act_dict[kc.AGENT_START_TIME]),
                                         typeID=kind)

    def step(self) -> tuple:
        """Advances the SUMO simulation by one timestep and retrieves information
        about vehicle arrivals and detector data.

        Returns:
            self.timestep (int): The current simulation timestep.
            arrivals (list): List of vehicle IDs that arrived at their destinations during the current timestep.
        """
   
        arrivals = self.sumo_connection.simulation.getArrivedIDList()
        self.sumo_connection.simulationStep()
        self.timestep += 1
        
        return self.timestep, arrivals