from bs4 import BeautifulSoup
import heapq
import networkx as nx
import pandas as pd
import traci
import xml.etree.ElementTree as ET
from queue import PriorityQueue



from human_learning import logit
from human_learning import gawron
from keychain import Keychain as kc
from services import path_generator
from services import remove_double_quotes



class Simulator:

    """
    The interface between traffic simulator (SUMO, HihgEnv, Flow) and the environment
    """

    def __init__(self, agents, params): #, connection_file_path, edge_file_path, route_file_path, empty_route_1, empty_route_2, beta, nr_iterations):
        #self.connection_file_path = connection_file_path
        #self.edge_file_path = edge_file_path
        #self.route_file_path = route_file_path
        #self.empty_route_1 = empty_route_1
        #self.empty_route_2 = empty_route_2
        #self.beta = beta
        #self.nr_iterations = nr_iterations


        self.sumo_type = params[kc.SUMO_TYPE]
        self.config = params[kc.SUMO_CONFIG_PATH]

        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS]
        self.beta = params[kc.BETA]

        # network x graph
        connection_file = params[kc.CONNECTION_FILE_PATH]
        edge_file = params[kc.EDGE_FILE_PATH]
        route_file = params[kc.ROUTE_FILE_PATH]
        self.G = self.network(connection_file, edge_file, route_file)
        self.routes = {}
        # The network is build on a OSM map
        
        # initialize the routes (like google maps) 
        ## beta ?
        ## put the parameters of origin and dest in the params
        
        origin1, origin2 = params[kc.ORIGIN1], params[kc.ORIGIN2]
        destination1, destination2 = params[kc.DESTINATION1], params[kc.DESTINATION2]

        origins = [origin1, origin2]
        destinations = [destination1, destination2]

        ### case that all the origins are connected with all the destinations
        """
        for origin, destination in itertools.product(origins, destinations):
            # Call find_best_paths for each combination of origin and destination
            result = self.find_best_paths(origin, destination, 'time')
            
            # Store the result in the self.routes dictionary
            self.routes[(origin, destination)] = result """
        
        for origin, destination in zip(origins, destinations):
            # Call find_best_paths for each combination of origin and destination
            result = self.find_best_paths(origin, destination, 'time')
            
            # Store the result in the self.routes dictionary
            self.routes[(origin, destination)] = result

        print(self.routes.keys())

        self.route1 = self.find_best_paths(origin1, destination1, 'time') ### self.routes
        self.route2 = self.find_best_paths(origin2, destination2, 'time') ## dict and items ->od combinations

        self.csv=pd.read_csv("agents_data.csv")

        csv1= self.csv[self.csv.origin==0]
        csv2= self.csv[self.csv.origin==1]

        # free flow travel time
        # maybe not for machines ???
        ###old version - must go, just needed to run_simulatio
        self.cost1 = self.free_flow_time(self.route1, csv1) 
        self.cost2 = self.free_flow_time(self.route2, csv2)

        #self.free_flow_cost = self.free_flow_time(self.route1, csv1) + self.free_flow_time(self.route2, csv2) 
        #print(self.cost1, self.cost2)

        ## Init returns None
        


    def read_joint_actions_df(self, joint_action_df):
        pass

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

            length.append(sum(rou))

        return length
    
    def calculate_free_flow_time(self):
        length=pd.DataFrame(self.G.edges(data=True))

        time=length[2].astype('str').str.split(':',expand=True)[1]
        length[2]=time.str.replace('}','',regex=True).astype('float')

        in_time1=self.free_flow_time_finder(self.route1,length[0],length[1],length[2])
        in_time2=self.free_flow_time_finder(self.route2,length[0],length[1],length[2])

        return in_time1 + in_time2
    
    ###old version - added only to run simulation because of cost1/cost2
    def free_flow_time(self, route,csv):
        length=pd.DataFrame(self.G.edges(data=True))
        time=length[2].astype('str').str.split(':',expand=True)[1]
        length[2]=time.str.replace('}','',regex=True).astype('float')
        in_time=self.free_flow_time_finder(route,length[0],length[1],length[2])
        cost=[]
        for _ in range(len(csv)):
            cost.append(in_time)
        column_names = [f"cost{i+1}" for i in range(len(cost[0]))]
        cost_df = pd.DataFrame(cost, columns=column_names)

        return cost_df
    


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
    
    def run_simulation_iteration(self, joint_action, csv):#This is the simulation where we use the values from the initial simulation to improve the initial solutions of te cars
        #### joint action - columns{id, origin, destination, actions, start_time}
        #### queue ordered by start_time

        sorted_rows = self.priority_queue_creation(joint_action)
        sorted_df = pd.DataFrame(sorted_rows, columns=pd.DataFrame(joint_action).columns)
        print(sorted_df)

        # Start SUMO with TraCI
        csv=pd.read_csv(csv)
        counter=csv.start_time.value_counts().sort_index() ### add the vehicles in a queue based on their start time
        csv1=csv[csv.origin==0]
        csv2=csv[csv.origin==1]
        
        sumo_binary = self.sumo_type
        sumo_cmd = [sumo_binary, "-c", self.config]

        traci.start(sumo_cmd)
        route_1_rou=[]
        route_1_veh=[]
        route_2_rou=[]
        route_2_veh=[]
        v=0

        try:
            # Set up demand (routes)
            for i in range(len(self.route1)):
                traci.route.add(f"route1_{i}", self.route1[i])
            for i in range(len(self.route2)):
                traci.route.add(f"route2_{i}", self.route2[i])

            # Simulation loop
            for x in range(self.simulation_length):
                traci.simulationStep()
                for y in range(len(csv1)):
                    if x==counter.index[y]:

                        cost1_in=list(self.cost1.iloc[v])
                        cost1_1=gawron(0.2,cost1_in,cost1_in)
                        self.cost1.iloc[v]=cost1_1
                        j=logit(self.beta,cost1_1) ## actions

                        cost2_in=list(self.cost2.iloc[v])
                        cost2_2=gawron(0.2,cost2_in,cost2_in)

                        self.cost2.iloc[v]=cost2_2
                        k=logit(self.beta,cost2_2) ##actions
                        vechicle_id1=f"{csv1.id.iloc[v]}"
                        vechicle_id2=f"{csv2.id.iloc[v]}"

                        traci.vehicle.add(vechicle_id1,f'route1_{j}')
                        traci.vehicle.add(vechicle_id2,f'route2_{k}')
                        traci.vehicle.setColor(vechicle_id2,(255,0,255))

                        route_1_rou.append(j)
                        route_1_veh.append(vechicle_id1)
                        route_2_rou.append(k)
                        route_2_veh.append(vechicle_id2)
                        v+=1

            # End of simulation
        finally:
            traci.close()

        
        route_1,route_2= self.travel_time('tripinfo.xml',route_1_rou,route_2_rou,csv1,csv2,self.cost1,self.cost2)
        self.cost1= self.time_update(route_1,self.cost1)
        time_route1=pd.merge(route_1,self.cost1,right_index=True,left_index=True)    
        self.cost2= self.time_update(route_2,self.cost2)
        time_route2=pd.merge(route_2,self.cost2,right_index=True,left_index=True)

        return time_route1,time_route2

    

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
            #path = self.routing(origin, destination, weight, picked_nodes)
            #while not path: path = self.routing(origin, destination, weight, picked_nodes)
            paths.append(path)
            picked_nodes.update(path)

        return paths
    


    def travel_time(self, file_path,route_1_rou,route_2_rou,csv1,csv2,cost1,cost2):
        id=pd.DataFrame(pd.read_xml(file_path).id).rename(columns={'id':'car_id'})

        dur=pd.read_xml(file_path).rename(columns={'duration':'cost'}).cost/60

        time=pd.merge(id,dur,right_index=True,left_index=True)
        route_1=pd.merge(csv1,time,left_on='id',right_on='car_id',how='left')

        route_1=route_1.drop(columns='car_id')
        route_2=pd.merge(csv2,time,left_on='id',right_on='car_id',how='left')

        route_2=route_2.drop(columns='car_id')

        rou1=pd.DataFrame(route_1_rou).rename(columns={0:'route_id'})
        rou2=pd.DataFrame(route_2_rou).rename(columns={0:'route_id'})

        route_1=pd.merge(route_1,rou1,right_index=True,left_index=True)
        route_2=pd.merge(route_2,rou2,right_index=True,left_index=True)

        route_1=self.replace(route_1,cost1)
        route_2=self.replace(route_2,cost2)

        return route_1,route_2
    


    
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
    
    def replace(self, times, zeros):

        for x in range(len(times)):
            for y in range(len(zeros)):

                if pd.isna(times.cost[y]) and times.index[x]==zeros.index[y]:
                        times.at[x,'cost']=zeros.iloc[:,times.route_id[x]][y]
        return times