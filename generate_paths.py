import networkx as nx
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from keychain import Keychain as kc
from utilities import get_params
from utilities import list_to_string
from utilities import remove_double_quotes
from utilities import df_to_prettytable


####### Network Generation


def generate_network(connection_file, edge_file, route_file):
    # Connection file
    from_db, to_db = _read_xml_file(connection_file, 'connection', 'from', 'to')
    from_to = pd.merge(from_db,to_db,left_index=True,right_index=True)
    from_to = from_to.rename(columns={'0_x':'From','0_y':'To'})
    # Edge file
    id_db, from_db = _read_xml_file(edge_file, 'edge', 'id', 'from')
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


def _read_xml_file(file_path, element_name, attribute_name, attribute_name_2):
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


####### Route Generation

def create_routes(number_of_paths, origins, destinations, beta, weight, num_samples=50, max_path_length=100):
    routes = dict()
    for dest_id, dest_sim_code in destinations.items():
        distances_to_destination = nx.single_source_dijkstra_path_length(traffic_graph, dest_sim_code, weight = weight)
        proximity = lambda x: distances_to_destination[x]
        for origin_id, origin_sim_code in origins.items():
            sampled_paths = set()
            while len(sampled_paths) < num_samples:
                path = _path_generator(traffic_graph, origin_sim_code, dest_sim_code, proximity, beta, max_path_length)
                sampled_paths.add(tuple(path))
            sampled_paths = list(sampled_paths)
            paths_idcs = np.random.choice(len(sampled_paths), number_of_paths, replace=False, p=_create_route_probabilities(sampled_paths)).tolist()
            routes[(origin_id, dest_id)] = [sampled_paths[idx] for idx in paths_idcs]
            print(f"[INFO] Generated {len(routes[(origin_id, dest_id)])} paths for {origin_id} -> {dest_id}")
    print(f"[SUCCESS] Generated {len(routes) * number_of_paths} routes")
    return routes


def _create_route_probabilities(sampled_paths):
    free_flows = [_get_ff(route) ** 2 for route in sampled_paths]
    prob1 = max(free_flows) - np.array(free_flows)
    prob1 = prob1 / np.sum(prob1)

    route_lengths = [len(route) ** 2 for route in sampled_paths]
    prob2 = max(route_lengths) - np.array(route_lengths)
    prob2 = prob2 / np.sum(prob2)

    prob = 0.5 * prob1 + 0.5 * prob2
    return prob



def _path_generator(network, origin, destination, proximity_func, beta, maxlen):
    path, current_node = list(), origin
    should_add_node = lambda node, path: (not ((node in path) or (node.removeprefix("-") in path) or (f"-{node}" in path))) or (node == destination)
    while True:
        path.append(current_node)
        options = [node for node in network.neighbors(current_node) if should_add_node(node, path)]
        if    destination in options:                return path + [destination]
        elif  (not options) or (len(path) > maxlen): return _path_generator(network, origin, destination, proximity_func, beta, maxlen)
        else:                                        current_node = _logit(options, proximity_func, beta)



def _logit(options, cost_function, beta):
    numerators = [np.exp(beta * cost_function(option)) for option in options]
    utilities = [numerator/sum(numerators) for numerator in numerators]
    return np.random.choice(options, p = utilities)


####### FF Times
    

def calculate_free_flow_times(od_paths_dict, show=False):
    length = pd.DataFrame(traffic_graph.edges(data = True))
    time = length[2].astype('str').str.split(':',expand=True)[1]
    length[2] = time.str.replace('}','',regex=True).astype('float')
    free_flows_dict = dict()
    for od, routes_of_od in od_paths_dict.items():
        free_flows=[]
        for route in range(len(routes_of_od)):
            rou=[]
            for i in range(len(routes_of_od[route])):
                if i < len(routes_of_od[route]) - 1:
                    for k in range(len(length[0])):
                        if (routes_of_od[route][i] == length[0][k]) and (routes_of_od[route][i + 1] == length[1][k]):
                            rou.append(length[2][k])
            free_flows.append(sum(rou))
        free_flows_dict[od] = free_flows
    if show: df_to_prettytable(pd.DataFrame(free_flows_dict), "FF Times")
    return free_flows_dict


def _get_ff(path):
    length = pd.DataFrame(traffic_graph.edges(data = True))
    time = length[2].astype('str').str.split(':',expand=True)[1]
    length[2] = time.str.replace('}','',regex=True).astype('float')
    rou=[]
    for i in range(len(path)):
        if i < len(path) - 1:
            for k in range(len(length[0])):
                if (path[i] == length[0][k]) and (path[i + 1] == length[1][k]):
                    rou.append(length[2][k])
    return sum(rou)


#######


def save_paths(routes, ff_times, paths_csv_save_path, routes_xml_save_path):
    # csv file, for us
    paths_df = pd.DataFrame(columns = [kc.ORIGIN, kc.DESTINATION, kc.PATH, kc.FREE_FLOW_TIME])
    for od, paths in routes.items():
        for path_idx, path in enumerate(paths):
            paths_df.loc[len(paths_df.index)] = [od[0], od[1], list_to_string(path, ","), ff_times[od][path_idx]]
    paths_df.to_csv(paths_csv_save_path, index=False)
    # XML file, for sumo
    with open(routes_xml_save_path, "w") as rou:
        print("""<routes>""", file=rou)
        for od, paths in routes.items():
                for idx, path in enumerate(paths):
                    print(f'<route id="{od[0]}_{od[1]}_{idx}" edges="',file=rou)
                    print(list_to_string(path,separator=' '),file=rou)
                    print('" />',file=rou)
        print("</routes>", file=rou)
    print(f"[SUCCESS] Saved {len(paths_df)} paths to: {paths_csv_save_path} and {routes_xml_save_path}")


#######


if __name__ == "__main__":
    params = get_params(kc.PARAMS_PATH)
    params = params[kc.PATH_GEN]

    number_of_paths = params[kc.NUMBER_OF_PATHS]
    beta = params[kc.BETA]
    weight = params[kc.WEIGHT]
    num_samples = params[kc.NUM_SAMPLES]
    max_path_length = params[kc.MAX_PATH_LENGTH]
    origins = params[kc.ORIGINS]
    destinations = params[kc.DESTINATIONS]

    connection_file_path = kc.CONNECTION_FILE_PATH
    edge_file_path = kc.EDGE_FILE_PATH
    route_file_path = kc.ROUTE_FILE_PATH
    paths_csv_save_path = kc.PATHS_CSV_SAVE_PATH
    routes_xml_save_path = kc.ROUTES_XML_SAVE_PATH

    origins = {i : origin for i, origin in enumerate(origins)}
    destinations = {i : dest for i, dest in enumerate(destinations)}

    traffic_graph = generate_network(connection_file_path, edge_file_path, route_file_path)
    routes = create_routes(number_of_paths, origins, destinations, beta, weight, num_samples, max_path_length)
    ff_times = calculate_free_flow_times(routes, show=True)
    save_paths(routes, ff_times, paths_csv_save_path, routes_xml_save_path)