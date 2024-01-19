from bs4 import BeautifulSoup
import networkx as nx
import pandas as pd
import traci
import xml.etree.ElementTree as ET
from Decision_model import logit


class Simulator:

    """
    The interface between traffic simulator (SUMO, HihgEnv, Flow) and the environment
    """

    def __init__(self): #, connection_file_path, edge_file_path, route_file_path, empty_route_1, empty_route_2, beta, nr_iterations):
        #self.connection_file_path = connection_file_path
        #self.edge_file_path = edge_file_path
        #self.route_file_path = route_file_path
        #self.empty_route_1 = empty_route_1
        #self.empty_route_2 = empty_route_2
        #self.beta = beta
        #self.nr_iterations = nr_iterations

        # network x graph
        self.G = self.network('Network_and_config/csomor1.con.xml','Network_and_config/csomor1.edg.xml','Network_and_config/csomor1.rou.xml')# The network is build on a OSM map
        
        # initialize the routes (like google maps) 
        ## beta ?
        self.route1 = self.iterate('279952229#0','-115602933#2', 'time',list(), -0.1,3)#I selected two source and target randomly, but here we can define anything else
        self.route2 = self.iterate('115604053','-441496282#1', 'time', list(),-0.1,3)

        # free flow travel time
        # maybe not for machines ???
        self.in_time1, self.in_time2 = self.free_flow_time() 

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

    def free_flow_time(self):
        length = pd.DataFrame(self.G.edges(data=True))
        time = length[2].astype('str').str.split(':',expand=True)[1]
        length[2] = time.str.replace('}','',regex=True).astype('float')

        #### Changed
        in_time1 = self.free_flow_time_finder(self.route1, length[0], length[1], length[2])
        in_time2 = self.free_flow_time_finder(self.route2, length[0], length[1], length[2])

        return in_time1, in_time2
    


    def network(self, connection_file, edge_file, route_file):
        # Connection file
        from_db, to_db = self.read_xml_file(connection_file, 'connection', 'from', 'to')
        from_to = pd.merge(from_db,to_db,left_index=True,right_index=True)
        from_to = from_to.rename(columns={'0_x':'From','0_y':'To'})
        
        # Edge file
        id_db, from_db = self.read_xml_file(edge_file, 'edge', 'id', 'from')

        id_name = pd.merge(from_db,id_db,right_index=True,left_index=True)

        id_name['0_x']=[self.stringer(x) for x in id_name['0_x']]
        id_name['0_y']=[self.stringer(x) for x in id_name['0_y']]
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



    

    def create_network_from_xml(self, connection_file, edge_file, route_file):
        # Connection file
        from_db, to_db = self.read_xml_file(connection_file, 'connection', 'from', 'to')
        from_to = pd.merge(from_db,to_db,left_index=True,right_index=True)
        from_to = from_to.rename(columns={'0_x':'From','0_y':'To'})
        
        # Edge file
        id_db, from_db = self.read_xml_file(edge_file, 'edge', 'id', 'from')

        id_name = pd.merge(from_db,id_db,right_index=True,left_index=True)

        id_name['0_x']=[self.stringer(x) for x in id_name['0_x']]
        id_name['0_y']=[self.stringer(x) for x in id_name['0_y']]
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




    def run_simulation_initial(self, sumo_type, config, route1, route2, j, k, simulation_length):
        # This is the initial simulation. This create the initial travel times and all the later used variables
        # Start SUMO with TraCI
        sumo_binary = sumo_type
        sumo_cmd = [sumo_binary, "-c", config]
        traci.start(sumo_cmd)
        route_1_rou=[]
        route_1_veh=[]
        route_2_rou=[]
        route_2_veh=[]

        try:
            # Set up demand (routes), the route creation for the simulation
            for i in range(len(route1)):
                traci.route.add(f"route1_{i}", route1[i])
            for i in range(len(route2)):
                traci.route.add(f"route2_{i}", route2[i])

            # Simulation loop
            for _ in range(simulation_length):#the simulation itself, 3600 timesteps
                traci.simulationStep()
                if _%6==0:
                    vechicle_id1=f"vehicle1_{_}"
                    vechicle_id2=f"vehicle2_{_}"
                    traci.vehicle.add(vechicle_id1,f'route1_{j}')
                    traci.vehicle.add(vechicle_id2,f'route2_{k}')
                    traci.vehicle.setColor(vechicle_id2,(255,0,255))
                    route_1_rou.append(j)
                    route_1_veh.append(vechicle_id1)
                    route_2_rou.append(k)
                    route_2_veh.append(vechicle_id2)
                    # Retrieve information using TraCI functions
                    # For example, get vehicle positions, routes, travel times, etc.

                # End of simulation
        finally:
            traci.close()

        df1,df2 = None, None #travel_time('tripinfo.xml',route_1_rou,route_2_rou,route_1_veh,route_2_veh)#ezt hogy azt adja ki

        return df1,df2
    
    def routing(self, origin, destination, weight, route, beta):
        origin_1=origin
        route1=[origin]
        lent=nx.single_source_dijkstra_path_length(self.G, origin_1, weight = weight) 

        while origin_1!=destination:
            
            log_keys=list(self.G.neighbors(origin_1))

            log_key = list(filter(lambda k: bool(list(self.G.neighbors(k))) or k==destination, log_keys))

            log=list(map(lambda n: route1.append(n) if n == destination else 0.0000000001 if n in route1 else 0.01 if any(map(lambda sublist: n in sublist, route)) else lent[n], log_key))

            if destination in route1:
                    break

            #time=time_generator(log)
            choosen=logit(beta,log)
            route1.append(log_key[choosen])
            origin_1=log_key[choosen]
        
        return route1


    def iterate(self, origin, destination, weight, route, beta, n):
        paths=[]

        for i in range(n):
            paths.append(self.routing(origin, destination, weight, route, beta))
            route=paths

        return paths
    


    def stringer(self, x):
        x = str(x).replace('"','')
        return x
    
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