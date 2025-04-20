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

from concurrent.futures import ProcessPoolExecutor, as_completed

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
        using_custom_demand (bool):
            Flag to indicate whether user provides custom travel demand data.
        
    Attributes:
        network_name: Network name.
        simulation_length: Simulation length.
        sumo_id: SUMO connection id.
        sumo_connection: Traci-SUMO connection object.
        timestep: Time step being simulated within the day.
    """

    def __init__(self, params: dict, path_gen_params: dict, seed: int = 23423, using_custom_demand: bool = False, save_detectors_info : bool = False) -> None:
        self.network_name        = params[kc.NETWORK_NAME]
        self.sumo_type           = params[kc.SUMO_TYPE]
        self.number_of_paths     = params[kc.NUMBER_OF_PATHS]
        self.simulation_length   = params[kc.SIMULATION_TIMESTEPS]
        self.stuck_time          = params[kc.STUCK_TIME]

        if self.network_name in kc.NETWORK_NAMES:
            curr_dir = os.path.dirname(os.path.abspath(__file__))

            self.network_folder      = os.path.join(curr_dir,
                                                    kc.NETWORK_FOLDER).replace("$net$", self.network_name)
            self.network_file_path   = os.path.join(curr_dir,
                                                    kc.NETWORK_FILE_PATH).replace("$net$", self.network_name)
            self.routes_xml_path     = os.path.join(curr_dir,
                                                    kc.ROU_FILE_PATH).replace("$net$", self.network_name)
            self.conn_file_path      = os.path.join(curr_dir,
                                                    kc.CONNECTION_FILE_PATH).replace("$net$", self.network_name)
            self.edge_file_path      = os.path.join(curr_dir,
                                                    kc.EDGE_FILE_PATH).replace("$net$", self.network_name)
            self.nod_file_path       = os.path.join(curr_dir,
                                                    kc.NOD_FILE_PATH).replace("$net$", self.network_name)
        else:
            self.network_folder      = params[kc.CUSTOM_NETWORK_FOLDER] if params[kc.CUSTOM_NETWORK_FOLDER] != "NA" else self.network_name
            self.network_file_path   = os.path.join(self.network_folder, self.network_name + ".net.xml")
            self.routes_xml_path     = os.path.join(self.network_folder, self.network_name + ".rou.xml")
            self.conn_file_path      = os.path.join(self.network_folder, self.network_name + ".con.xml")
            self.edge_file_path      = os.path.join(self.network_folder, self.network_name + ".edg.xml")
            self.nod_file_path       = os.path.join(self.network_folder, self.network_name + ".nod.xml")
          
        self.det_xml_save_path       = os.path.join(params[kc.RECORDS_FOLDER], kc.DETECTORS_XML_FILE_NAME)
        self.paths_csv_file_path     = os.path.join(params[kc.RECORDS_FOLDER], kc.PATHS_CSV_FILE_NAME)
        self.rou_xml_save_path       = os.path.join(params[kc.RECORDS_FOLDER], kc.ROUTE_XML_FILE_NAME)
        self.sumo_save_path          = os.path.join(params[kc.RECORDS_FOLDER], kc.SUMO_LOGS_FOLDER)
        
        random.seed(seed)

        self.seed = seed
        self.sumo_id = f"{random.randint(0, 1000)}"
        self.sumo_connection = None
        self.save_detectors_info = save_detectors_info

        confirm_env_variable(kc.ENV_VAR, append="tools")

        if path_gen_params is not None:
            self._get_paths(params, path_gen_params, using_custom_demand)
            logging.info("[SUCCESS] Path generation completed.")
        self._check_paths_ready()
        self.detectors_name = self._get_detectors()
        
        self.timestep = 0
        self.runs = 0
        self.route_id_cache = dict()
        self.waiting_vehicles = dict()

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
            
    def _get_paths(self, params: dict, path_gen_params: dict, using_custom_demand: bool) -> None:

        # Build the network
        network = jx.build_digraph(self.conn_file_path, self.edge_file_path, self.routes_xml_path)
        
        # Get origins and destinations
        origins = path_gen_params[kc.ORIGINS]
        destinations = path_gen_params[kc.DESTINATIONS]
        
        # Get demand, if using custom demand
        if using_custom_demand:
            demand_df = pd.read_csv(os.path.join(params[kc.RECORDS_FOLDER], kc.AGENTS_CSV_FILE_NAME))
            demands = list(zip(demand_df[kc.AGENT_ORIGIN], demand_df[kc.AGENT_DESTINATION]))
            demands = list(set(demands))
        else:
            demands = None
        
        path_gen_kwargs = {
            "number_of_paths": path_gen_params[kc.NUMBER_OF_PATHS],
            "random_seed": self.seed,
            "num_samples": path_gen_params[kc.NUM_SAMPLES],
            "beta": path_gen_params[kc.BETA],
            "weight": path_gen_params[kc.WEIGHT],
            "verbose": False
        }
        
        if demands is None:
            routes = jx.basic_generator(network, origins, destinations, as_df=True, calc_free_flow=True, **path_gen_kwargs)
        else:
            routes = pd.DataFrame(columns=["origins", "destinations", "path", "free_flow_time"])
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._route_gen_process, network, demands, origins, destinations, idx, path_gen_kwargs) for idx in range(len(demands))]
                for i, future in enumerate(as_completed(futures), 1):
                    #print(f"\r{i}/{len(demands)} - {demands[i]}", end="")
                    routes_df = future.result()
                    routes = pd.concat([routes, routes_df], ignore_index=True)
            
        self._save_paths_to_disc(routes, origins, destinations)
        
        # Visualize paths and save figures
        if path_gen_params[kc.VISUALIZE_PATHS]:
            path_visuals_path = params[kc.PLOTS_FOLDER]
            os.makedirs(path_visuals_path, exist_ok=True)
            
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._route_vis_process, demands, origin_idx, dest_idx, origin, destination, routes, path_visuals_path) for origin_idx, origin in enumerate(origins) for dest_idx, destination in enumerate(destinations)]
            """
            for origin_idx, origin in enumerate(origins):
                for dest_idx, destination in enumerate(destinations):
                    if (demands is not None) and (not (origin_idx, dest_idx) in demands):
                        continue
                    # Filter routes for the current origin-destination pair
                    routes_to_show = (routes[(routes["origins"] == origin_idx)
                                            & (routes["destinations"] == dest_idx)]['path'])
                    routes_to_show = [route.split(" ") for route in routes_to_show]
                    # Specify the save path and title for the figure
                    fig_save_path = os.path.join(path_visuals_path, f"{origin_idx}_{dest_idx}.png")
                    title=f"Origin: {origin_idx} ({origin}), Destination: {dest_idx} ({destination})"
                    # Show the routes
                    try:
                        jx.show_multi_routes(self.nod_file_path, self.edge_file_path,
                                            routes_to_show, origin, destination, 
                                            show=False, save_file_path=fig_save_path, title=title)
                    except:
                        logging.warning(f"Could not visualize routes for {origin} to {destination}.")
            """
                        
                        
    def _route_gen_process(self, network, demands, origins, destinations, demand_idx, path_gen_kwargs):
        origin = origins[demands[demand_idx][0]]
        destination = destinations[demands[demand_idx][1]]
        return jx.extended_generator(
            network=network,
            origins=[origin],
            destinations=[destination],
            as_df=True,
            calc_free_flow=True,
            **path_gen_kwargs
        )
        
    def _route_vis_process(self, demands, origin_idx, dest_idx, origin, destination, routes, path_visuals_path):
        if (demands is not None) and (not (origin_idx, dest_idx) in demands):
            return
        # Filter routes for the current origin-destination pair
        routes_to_show = (routes[(routes["origins"] == origin_idx)
                                & (routes["destinations"] == dest_idx)]['path'])
        routes_to_show = [route.split(" ") for route in routes_to_show]
        # Specify the save path and title for the figure
        fig_save_path = os.path.join(path_visuals_path, f"{origin_idx}_{dest_idx}.png")
        title=f"Origin: {origin_idx} ({origin}), Destination: {dest_idx} ({destination})"
        # Show the routes
        try:
            jx.show_multi_routes(self.nod_file_path, self.edge_file_path,
                                routes_to_show, origin, destination, 
                                show=False, save_file_path=fig_save_path, title=title)
        except:
            logging.warning(f"Could not visualize routes for {origin} to {destination}.")

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
        
        with open(self.det_xml_save_path, "w") as det:
            print("""<additional>""", file=det)
            for det_id in detectors_name:
                print(f"<inductionLoop id=\"{det_id}_det\" lane=\"{det_id}_0\" pos=\"-5\" file=\"NUL\" friendlyPos=\"True\"/>", file=det)
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
        
        self.runs += 1

        individual_sumo_stats_file = os.path.join(self.sumo_save_path,
                                                  f"detailed_sumo_stats_{self.runs}.xml")
        
        combined_sumo_stats_file = os.path.join(self.sumo_save_path,
                                                f"sumo_stats_{self.runs}.xml")

        sumo_cmd = [
            self.sumo_type,
            "--seed",
            str(self.seed),
            "--net-file",
            self.network_file_path,
            "--additional-files",
            f"{self.det_xml_save_path},{self.rou_xml_save_path}",
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
            "--statistic-output",
            combined_sumo_stats_file,
            "--tripinfo-output",
            individual_sumo_stats_file
            ]
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

        det_dict = {name: 0 for name in self.detectors_name}
        if self.save_detectors_info:
            for det_name in self.detectors_name:
                det_dict[det_name]  = self.sumo_connection.lanearea.getIntervalVehicleNumber(f"{det_name}_det")

        self.runs += 1

        individual_sumo_stats_file = os.path.join(self.sumo_save_path,
                                                  f"detailed_sumo_stats_{self.runs}.xml")
        
        combined_sumo_stats_file = os.path.join(self.sumo_save_path,
                                                f"sumo_stats_{self.runs}.xml")
        
        sumo_cmd = [
            "--seed",
            str(self.seed),
            "--net-file",
            self.network_file_path,
            "--additional-files",
            f"{self.det_xml_save_path},{self.rou_xml_save_path}",
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
            "--statistic-output",
            combined_sumo_stats_file,
            "--tripinfo-output",
            individual_sumo_stats_file
            ]
        
        self.sumo_connection.load(sumo_cmd)
        

        self.timestep = 0
        self.waiting_vehicles = dict()
        return det_dict

    ################################
    ######### SIMULATION ###########
    ################################

    def retrieve_detector_data(self) -> None:
        """Return information about whether an a vehicle stopped in a detector."""
        self.stopped_vehicles_info = []

        for det_name in self.detectors_name:
            det_id = f"{det_name}_det"
            veh_ids = traci.lanearea.getLastStepVehicleIDs(det_id)

            for veh_id in veh_ids:
                speed = traci.vehicle.getSpeed(veh_id)
                if speed < 0.1:
                    self.stopped_vehicles_info.append({
                    "time": self.timestep,
                    "detector": det_id,
                    "vehicle_id": veh_id
                })
                    

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
        self.waiting_vehicles[str(act_dict[kc.AGENT_ID])] = 0

    def step(self) -> tuple:
        """Advances the SUMO simulation by one timestep and retrieves information
        about vehicle arrivals and detector data.

        Returns:
            self.timestep (int): The current simulation timestep.
            arrivals (list): List of vehicle IDs that arrived at their destinations during the current timestep.
        """
   
        arrivals = list(self.sumo_connection.simulation.getArrivedIDList())
        for arr in arrivals:
            self.waiting_vehicles.pop(arr, None)
        
        # Teleport vehicles that are stuck
        teleported = list()
        for veh_id in self.waiting_vehicles.copy():
            if self.sumo_connection.vehicle.getSpeed(veh_id) == 0:
                self.waiting_vehicles[veh_id] += 1
                if self.waiting_vehicles[veh_id] > self.stuck_time:
                    self.sumo_connection.vehicle.remove(veh_id)
                    logging.info(f"Timestep #{self.timestep}: Teleporting {veh_id} due to being stuck for {self.waiting_vehicles[veh_id]} seconds.")
                    teleported.append(veh_id)
                    self.waiting_vehicles.pop(veh_id, None)
            else:
                self.waiting_vehicles[veh_id] = 0
                
        # Advance the simulation by one timestep       
        self.sumo_connection.simulationStep()

        # Retrieve information about the detectors
        if self.save_detectors_info == True:
            self.retrieve_detector_data()
        else:
            self.stopped_vehicles_info = None

        self.timestep += 1
        
        return self.timestep, self.stopped_vehicles_info, arrivals, teleported