import networkx as nx
import pandas as pd
import traci
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from keychain import Keychain as kc
from services import SumoController
from utilities import list_to_string
from utilities import make_dir
from utilities import path_generator
from utilities import remove_double_quotes


class Simulator:

    """
    The interface between traffic simulator (SUMO, HihgEnv, Flow) and the environment
    """

    def __init__(self, params):

        self.sumo_controller = SumoController(params)
        self.routes_xml_save_path = params[kc.ROUTES_XML_SAVE_PATH]
        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS]
        self.beta = params[kc.BETA]

        # NX graph, built on a OSM map
        self.traffic_graph = self.generate_network(params[kc.CONNECTION_FILE_PATH], params[kc.EDGE_FILE_PATH], params[kc.ROUTE_FILE_PATH])

        # We keep origins and dests as dict {origin_id : origin_code}
        # Such as (0 : "279952229#0") and (1 : "279952229#0")
        # Keys are what agents know, values are what we use in SUMO
        self.origins = {i : origin for i, origin in enumerate(params[kc.ORIGINS])}
        self.destinations = {i : dest for i, dest in enumerate(params[kc.DESTINATIONS])}

        # We keep routes as dict {(origin_id, dest_id) : [list of nodes]}
        # Such as ((0,0) : [list of nodes]) and ((1,0) : [list of nodes])
        # In list of nodes, we use SUMO simulation ids of nodes
        self.routes = self.create_routes(self.origins, self.destinations)
        self.save_paths(self.routes)

        self.last_simulation_duration = 0

        print("[SUCCESS] Simulator is ready to simulate!")



    def start_sumo(self):
        self.sumo_controller.sumo_start()

    def stop_sumo(self):
        self.sumo_controller.sumo_stop()

    def reset_sumo(self):
        self.sumo_controller.sumo_reset()


    
    def get_last_sim_duration(self):
        return self.last_simulation_duration

        

    def save_paths(self, routes):
        # csv file, for us
        paths_df = pd.DataFrame(columns = ["origin", "destination", "path"])
        for od, paths in routes.items():
            for path in paths:
                paths_df.loc[len(paths_df.index)] = [od[0], od[1], list_to_string(path, "-> ")]
        save_to = make_dir(kc.RECORDS_FOLDER, kc.PATHS_CSV_FILE_NAME)
        paths_df.to_csv(save_to, index=True)
        print("[SUCCESS] Generated & saved %d paths to: %s" % (len(paths_df), save_to))
        
        # XML file, for sumo
        with open(self.routes_xml_save_path, "w") as rou:
            print("""<routes>""", file=rou)
            for od, paths in routes.items():
                    for idx, path in enumerate(paths):
                        print(f'<route id="{od[0]}_{od[1]}_{idx}" edges="',file=rou)
                        print(list_to_string(path,separator=' '),file=rou)
                        print('" />',file=rou)
            print("</routes>", file=rou)
        


    def create_routes(self, origins, destinations):
        routes = dict()
        for origin_id, origin_sim_code in origins.items():
            for dest_id, dest_sim_code in destinations.items():
                route = self.find_best_paths(origin_sim_code, dest_sim_code, 'time')
                routes[(origin_id, dest_id)] = route
        print(f"[SUCCESS] Generated {len(routes)} routes")
        return routes
    


    def find_best_paths(self, origin, destination, weight):
        paths = list()
        picked_nodes = set()

        for _ in range(self.number_of_paths):
            while True:
                path = path_generator(self.traffic_graph, origin, destination, weight, picked_nodes, self.beta)
                if path not in paths: break     # if path is not already generated, then break
            paths.append(path)
            picked_nodes.update(path)

        return paths



    def free_flow_time_finder(self, x, y, z, l):
        length=[]
        for route in range(len(x)):
            rou=[]
            for i in range(len(x[route])):
                if i < len(x[route]) - 1:
                    for k in range(len(y)):
                        if x[route][i] == y[k] and x[route][i + 1] == z[k]:
                            rou.append(l[k])
            length.append(sum(rou))

        return length
    


    def calculate_free_flow_times(self):
        length = pd.DataFrame(self.traffic_graph.edges(data = True))
        time = length[2].astype('str').str.split(':',expand=True)[1]
        length[2] = time.str.replace('}','',regex=True).astype('float')

        free_flows_dict = dict()
        # Loop through the values in self.routes
        for od, route in self.routes.items():
            # Call free_flow_time_finder for each route
            free_flow = self.free_flow_time_finder(route, length[0], length[1], length[2])
            # Append the free_flow value to the list
            free_flows_dict[od] = free_flow

        return free_flows_dict



    def generate_network(self, connection_file, edge_file, route_file):
        # Connection file
        from_db, to_db = self.read_xml_file(connection_file, 'connection', 'from', 'to')
        from_to = pd.merge(from_db,to_db,left_index=True,right_index=True)
        from_to = from_to.rename(columns={'0_x':'From','0_y':'To'})
        
        # Edge file
        id_db, from_db = self.read_xml_file(edge_file, 'edge', 'id', 'from')

        id_name = pd.merge(from_db,id_db,right_index=True,left_index=True)

        id_name['0_x']=[remove_double_quotes(x) for x in id_name['0_x']]
        id_name['0_y']=[remove_double_quotes(x) for x in id_name['0_y']]
        id_name=id_name.rename(columns={'0_x':'Name','0_y':'ID'})
        
        # Route file
        with open(route_file, 'r') as f:
            data_rou = f.read()
        Bs_data_rou = BeautifulSoup(data_rou, "xml")

        # Extract <connection> elements with 'via' attribute
        rou = Bs_data_rou.find_all('edge', {'to': True})

        empty = [str(rou[x]) for x in range(len(rou))]

        id, length, speed = list(), list(), list()
        for x in range(len(empty)):
            root = ET.fromstring(empty[x])
            id.append(root.attrib.get('id'))
            length.append(root.find('.//lane').attrib.get('length'))
            speed.append(root.find('.//lane').attrib.get('speed'))
        
        id_db=pd.DataFrame(id)
        len_db=pd.DataFrame(length)
        speed_db=pd.DataFrame(speed)

        speed_name=pd.merge(speed_db,id_db,right_index=True,left_index=True)
        speed_name=speed_name.rename(columns={'0_x':'speed','0_y':'ID'})

        len_name=pd.merge(len_db,id_db,right_index=True,left_index=True)
        len_name=len_name.rename(columns={'0_x':'length','0_y':'ID'})

        id_name=pd.merge(len_name,id_name,right_on='ID',left_on='ID')
        id_name=pd.merge(speed_name,id_name,right_on='ID',left_on='ID')

        final=pd.merge(id_name,from_to,right_on='From',left_on='ID')
        final=final.drop(columns=['ID'])
        final=pd.merge(id_name,final,right_on='To',left_on='ID')
        final['time']=((final['length_x'].astype(float)/(final['speed_x'].astype(float)))/60)
        final=final.drop(columns=['ID','length_y','speed_y','speed_x','length_x'])
        traffic_graph = nx.from_pandas_edgelist(final, 'From', 'To', ['time'], create_using=nx.DiGraph())
        return traffic_graph
    


    def joint_action_to_sorted_stack(self, joint_action):
        # Sort the joint_action dataframe by start times (descending order for stack)
        sorted_joint_action = joint_action.sort_values(kc.AGENT_START_TIME, ascending=False)

        # Make a sumo_action column in sorted_joint_action dataframe
        sumonize_action = lambda row: f'{row[kc.AGENT_ORIGIN]}_{row[kc.AGENT_DESTINATION]}_{row[kc.ACTION]}'
        sorted_joint_action[kc.SUMO_ACTION] = sorted_joint_action.apply(sumonize_action, axis=1)

        # Create a stack of agents and their sumo actions
        stack_bottom_placeholder = {kc.AGENT_START_TIME : -1}
        agents_stack = [stack_bottom_placeholder]

        for _, row in sorted_joint_action.iterrows():
            stack_row = {kc.AGENT_ID : f"{row[kc.AGENT_ID]}", kc.AGENT_START_TIME : row[kc.AGENT_START_TIME], kc.SUMO_ACTION : row[kc.SUMO_ACTION]}
            agents_stack.append(stack_row)

        return agents_stack
    


    def run_simulation_iteration(self, joint_action):
        arrivals = {kc.AGENT_ID : list(), kc.ARRIVAL_TIME: list()}  # Where we save arrivals
        agents_stack = self.joint_action_to_sorted_stack(joint_action)  # Where we keep agents and their actions
        should_continue = True

        # Simulation loop
        while should_continue:
            timestep = int(traci.simulation.getTime())

            # Add vehicles to the simulation
            while agents_stack[-1][kc.AGENT_START_TIME] == timestep:
                row = agents_stack.pop()
                traci.vehicle.add(row[kc.AGENT_ID], row[kc.SUMO_ACTION])

            # Collect vehicles that have reached their destination
            arrived_now = traci.simulation.getArrivedIDList()   # returns a list of arrived vehicle ids
            arrived_now = [int(value) for value in arrived_now]   # Convert values to int

            for id in arrived_now:
                arrivals[kc.AGENT_ID].append(id)
                arrivals[kc.ARRIVAL_TIME].append(timestep)
            
            # Did all vehicles arrive?
            should_continue = len(arrivals[kc.AGENT_ID]) < len(joint_action)
            # Advance the simulation
            traci.simulationStep()
        
        # Needed for plots
        self.last_simulation_duration = timestep
        # Calculate travel times
        travel_times_df = self.prepare_travel_times_df(arrivals, joint_action)
        return travel_times_df
        


    def prepare_travel_times_df(self, arrivals, joint_action):
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
        travel_times_df = travel_times_df[[kc.AGENT_ID, kc.TRAVEL_TIME]]

        return travel_times_df


    
    def read_xml_file(self, file_path, element_name, attribute_name, attribute_name_2):
        with open(file_path, 'r') as f:
            data = f.read()
        Bs_data_con = BeautifulSoup(data, "xml")
        
        connections = Bs_data_con.find_all(element_name)

        empty=[]
        for x in range(len(connections)):
            empty.append(str(connections[x]))

        from_=[]
        to_=[]
        for x in range(len(empty)):
            root = ET.fromstring(empty[x])
            from_.append(root.attrib.get(attribute_name))
            to_.append(root.attrib.get(attribute_name_2))

        from_db=pd.DataFrame(from_)
        to_db=pd.DataFrame(to_)
        return from_db, to_db
    
