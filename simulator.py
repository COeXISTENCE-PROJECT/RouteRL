from bs4 import BeautifulSoup
import networkx as nx
import pandas as pd
import traci
import xml.etree.ElementTree as ET
from queue import PriorityQueue


from keychain import Keychain as kc
from services import path_generator
from services import cursed_path_generator
from services import list_to_string
from services import remove_double_quotes



class Simulator:

    """
    The interface between traffic simulator (SUMO, HihgEnv, Flow) and the environment
    """

    def __init__(self, params):

        #self.sumo_type = params[kc.SUMO_TYPE]
        #self.config = params[kc.SUMO_CONFIG_PATH]

        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS]
        self.beta = params[kc.BETA]

        connection_file = params[kc.CONNECTION_FILE_PATH]
        edge_file = params[kc.EDGE_FILE_PATH]
        route_file = params[kc.ROUTE_FILE_PATH]

        self.G = self.network(connection_file, edge_file, route_file)   # network x graph
        # The network is build on a OSM map

        self.routes = dict()

        self.names=[]
        
        self.origin1, self.origin2 = params[kc.ORIGIN1], params[kc.ORIGIN2]
        self.destination1, self.destination2 = params[kc.DESTINATION1], params[kc.DESTINATION2]

        #origins = [self.origin1, self.origin2]
        #destinations = [self.destination1, self.destination2]
        origins=params[kc.ORIGIN]
        destinations=params[kc.DESTINATION]

        self.route_counter=[]

        ### case that all the origins are connected with all the destinations
        """
        for origin, destination in itertools.product(origins, destinations):
            # Call find_best_paths for each combination of origin and destination
            result = self.find_best_paths(origin, destination, 'time')
            
            # Store the result in the self.routes dictionary
            self.routes[(origin, destination)] = result """
        
        
        for origin, destination in zip(origins, destinations):
            # Call find_best_paths for each combination of origin and destination
            paths = self.find_best_paths(origin, destination, 'time')
            # Store the result in the self.routes dictionary
            self.routes[(origin,destination)] = paths   
        self.save_paths(self.routes, params)

        
    def save_paths(self, routes, params):

        path_attributes = ["origin", "destination", "path"]
        path_save_path = "paths.csv"
        
        origins=params[kc.ORIGIN]
        destinations=params[kc.DESTINATION]
        origin_df=pd.DataFrame(origins, columns=['origins'])
        destination_df=pd.DataFrame(destinations, columns=['destinations'])

        paths_df = pd.DataFrame(columns=path_attributes)

        for od, paths in routes.items():
            for path in paths:
                paths_df.loc[len(paths_df)] = [od[0], od[1], list_to_string(path, "-> ")]
        
        with open("Network_and_config/route.rou.xml", "w") as rou:
            print("""<routes>""", file=rou)
            for od, paths in routes.items():
                    list(od)[0]
                    ori=origin_df[origin_df['origins']==list(od)[0]].index.values[0]
                    dest=destination_df[destination_df['destinations']==list(od)[1]].index.values[0]
                    i=0
                    for path in paths:
                        print(f'<route id="{ori}_{dest}_{i}" edges="',file=rou)
                        self.names.append(f'{ori}_{dest}_{i}')
                        print(list_to_string(path,separator=' '),file=rou)
                        print('" />',file=rou)
                        i+=1
            print("</routes>", file=rou)

        paths_df.to_csv(path_save_path, index=True)
        print("[SUCCESS] Generated & saved %d paths to: %s" % (len(paths_df), path_save_path))


    def create_routes(self):
        # Will create action space (routes)
        pass


    def free_flow_time_finder(self, x, y, z, l):
        length=[]

        for route in range(len(x)):
            rou=[]

            for i in range(len(x[route])):
                if i < len(x[route]) - 1:
                    for k in range(len(y)):
                        if x[route][i] == y[k] and x[route][i + 1] == z[k]:
                            rou.append(l[k])

            #length.append(sum(rou))
            length.append(0)

        return length
    

    def calculate_free_flow_times(self):
        free_flows_dict = dict()
        for od in self.routes.keys():
            origin = 0 if od[0] == self.origin1 else 1
            destination = 0 if od[1] == self.destination1 else 1
            free_flows_dict[(origin, destination)] = list()

        length = pd.DataFrame(self.G.edges(data = True))

        time = length[2].astype('str').str.split(':',expand=True)[1]
        length[2] = time.str.replace('}','',regex=True).astype('float')

        # Loop through the values in self.routes
        for od, route in self.routes.items():

            # Call free_flow_time_finder for each route
            free_flow = self.free_flow_time_finder(route, length[0], length[1], length[2])
            
            # Append the in_time value to the list
            origin = 0 if od[0] == self.origin1 else 1
            destination = 0 if od[1] == self.destination1 else 1
            free_flows_dict[(origin, destination)] = free_flow

        return free_flows_dict


    def network(self, connection_file, edge_file, route_file):
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


        id=[]
        length=[]
        speed=[]
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
        final['time']=((final['length_x'].astype(float)/(final['speed_x'].astype(float)/3.6))/60)#np.exp
        final=final.drop(columns=['ID','length_y','speed_y','speed_x','length_x'])
        Graph = nx.from_pandas_edgelist(final, 'From', 'To', ['time'], create_using=nx.DiGraph())
    
        return Graph
    

    def priority_queue_creation(self, joint_action):
        #### joint action - columns{id, origin, destination, actions, start_time}
        #### queue ordered by start_time

        df = pd.DataFrame(joint_action)

        # Use heapq to create a priority queue
        priority_queue = PriorityQueue()

        # Iterate over the DataFrame and enqueue rows based on the sorting column
        for _, row in df.iterrows():
            priority_queue.put((row["start_time"], tuple(row)))

        sorted_rows = []
        while not priority_queue.empty():
            _, row = priority_queue.get()
            #print("row is: ", row, "\n")
            sorted_rows.append(row)

        return sorted_rows
    

    def run_simulation_iteration(self, joint_action):

        depart_id = []
        depart_cost = []
        self.route_counter = []

        sorted_rows_based_on_start_time = self.priority_queue_creation(joint_action)
        sorted_df = pd.DataFrame(sorted_rows_based_on_start_time, columns=pd.DataFrame(joint_action).columns)

        # Simulation loop
        for timestep in range(self.simulation_length):
            traci.simulationStep()
        
            if timestep==self.simulation_length-1:
                remove=traci.vehicle.getIDList()
                routes=traci.route.getIDList()

                for route in routes:
                    traci.simulation.clearPending(routeID=route)
                for i in remove:
                    traci.vehicle.remove(i,3)


            departed=traci.simulation.getArrivedIDList()  # just collect this and time and calculate at the end
            for value in departed:
                if value:
                    value_as_int = int(value)
                    depart_id.append(value_as_int)
                    start=sorted_df[sorted_df.id==value_as_int].start_time.values
                    depart_cost.append((timestep-start)/60)

            for _, row in sorted_df[sorted_df["start_time"] == timestep].iterrows():
                action = row["action"]
                vehicle_id = f"{row['id']}"
                ori=row['origin']
                dest=row['destination']
                traci.vehicle.add(vehicle_id, f'{ori}_{dest}_{action}')
                self.route_counter.append(f'{ori}_{dest}_{action}')


        depart=pd.DataFrame(depart_id)
        reward=pd.merge(depart,pd.DataFrame(depart_cost),right_index=True,left_index=True)
        reward=reward.rename(columns={'0_x':'car_id','0_y':'cost'})

        return reward
    

    def create_network_from_xml(self, connection_file, edge_file, route_file):
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

        empty=[]
        for x in range(len(rou)):
            empty.append(str(rou[x]))

        id=[]
        length=[]
        speed=[]
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
        final['time']=((final['length_x'].astype(float)/(final['speed_x'].astype(float)/3.6))/60)#np.exp
        final=final.drop(columns=['ID','length_y','speed_y','speed_x','length_x'])
        Graph = nx.from_pandas_edgelist(final, 'From', 'To', ['time'], create_using=nx.DiGraph())
        
        return Graph



    def find_best_paths(self, origin, destination, weight):
        paths = list()
        picked_nodes = set()

        for _ in range(self.number_of_paths):
            path = path_generator(self.G, origin, destination, weight, picked_nodes, self.beta)
            paths.append(path)
            picked_nodes.update(path)

        return paths

    
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
