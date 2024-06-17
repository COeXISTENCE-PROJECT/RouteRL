import networkx as nx
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from keychain import Keychain as kc
from utilities import df_to_prettytable
from utilities import get_params
from utilities import list_to_string
from utilities import remove_double_quotes


############# Network Generation ################

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

#################################################


############## OD Integrity #################

def check_od_integrity(network, origins, destinations):
    for dest_idx, destination in destinations.items():
        if not destination in network.nodes:    raise ValueError(f"Destination {dest_idx} is not in the network")
        distances_to_destination = dict(nx.shortest_path_length(network, target=destination))
        for origin_idx, origin in origins.items():
            if not origin in network.nodes:     raise ValueError(f"Origin {origin_idx} is not in the network")
            elif not origin in distances_to_destination:
                raise ValueError(f"Origin {origin_idx} cannot reach destination {dest_idx}")

#################################################


############## Route Generation #################

def create_routes(network, num_routes, origins, destinations, beta, weight, coeffs, num_samples=50, max_path_length=100):
    routes = dict()   # Tuple<od_id, dest_id> : List<routes>
    for dest_idx, dest_code in destinations.items():
        proximity_func = _get_proximity_function(network, dest_code, weight)   # Maps node -> proximity (cost)
        for origin_idx, origin_code in origins.items():
            sampled_routes = set()   # num_samples number of routes
            while len(sampled_routes) < num_samples:
                path = _path_generator(network, origin_code, dest_code, proximity_func, beta, max_path_length)
                if not path is None:    
                    sampled_routes.add(tuple(path))
                    print(f"\r[INFO] Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}", end="")
            routes[(origin_idx, dest_idx)] = _pick_routes_from_samples(sampled_routes, proximity_func, num_routes, coeffs, network)
            print(f"\n[INFO] Selected {len(routes[(origin_idx, dest_idx)])} paths for {origin_idx} -> {dest_idx}")
    return routes


def _get_proximity_function(network, destination, weight):
    # cost for all nodes that CAN access to destination
    distances_to_destination = dict(nx.shortest_path_length(network, target=destination, weight=weight))
    # dead-end nodes have infinite cost
    dead_nodes = [node for node in network.nodes if node not in distances_to_destination]
    for node in dead_nodes:  distances_to_destination[node] = float("inf")
    # return the lambda function
    return lambda x: distances_to_destination[x]


def _pick_routes_from_samples(sampled_routes, proximity, num_paths, coeffs, network):
    sampled_routes = list(sampled_routes)
    # what we base our selection on
    utility_dist = _get_route_utilities(sampled_routes, proximity, coeffs, network)
    # route indices that maximize defined utilities
    sorted_indices = np.argsort(utility_dist)[::-1]
    paths_idcs = sorted_indices[:num_paths]
    return [sampled_routes[idx] for idx in paths_idcs]


def _get_route_utilities(sampled_routes, proximity_func, coeffs, network):
    # Based on FF times
    free_flows = [_get_ff(route, network) for route in sampled_routes]
    utility1 = 1 / np.array(free_flows)
    utility1 = utility1 / np.sum(utility1)

    # Based on number of edges
    route_lengths = [len(route) for route in sampled_routes]
    utility2 = 1 / np.array(route_lengths)
    utility2 = utility2 / np.sum(utility2)

    # Based on proximity increase in consecutive nodes (how well & steady)
    prox_increase = [[proximity_func(route[idx-1]) - proximity_func(node) for idx, node in enumerate(route[1:])] for route in sampled_routes]
    mean_prox_increase = [np.mean(prox) for prox in prox_increase]
    std_prox_increase = [np.std(prox) for prox in prox_increase]
    utility3 = [mean_prox_increase[i] / std_prox_increase[i] for i in range(len(sampled_routes))]
    utility3 = np.array(utility3) / np.sum(utility3)
    
    # Based on uniqueness of the route (how different from other routes)
    lcs_values = [[lcs_non_consecutive(route, route2) for route2 in sampled_routes if route2 != route] for route in sampled_routes]
    lcs_values = [np.mean(lcs) for lcs in lcs_values]
    utility4 = 1 / np.array(lcs_values)
    utility4 = utility4 / np.sum(utility4)

    # Merge all with some coefficients
    utilities = (coeffs[0] * utility1) + (coeffs[1] * utility2) + (coeffs[2] * utility3) + (coeffs[3] * utility4)
    return utilities


def _path_generator(network, origin, destination, proximity_func, beta, maxlen):
    path, current_node = list(), origin
    while True:
        path.append(current_node)
        options = [node for node in network.neighbors(current_node) if node not in path]
        if   (destination in options):                  return path + [destination]
        elif (not options) or (len(path) > maxlen):     return None
        else:                                           current_node = _logit(options, proximity_func, beta)


def _logit(options, cost_function, beta):
    numerators = [np.exp(beta * cost_function(option)) for option in options]
    utilities = [numerator/sum(numerators) for numerator in numerators]
    return np.random.choice(options, p=utilities)


def lcs_non_consecutive(X, Y):
    """
    The LCS of two sequences is the longest subsequence that is present in both sequences in the same order, 
    but not necessarily consecutively.
    """
    m, n = len(X), len(Y)
    L = [[None]*(n+1) for i in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]


def lcs_consecutive(X, Y):
    """
    The LCS of two sequences is the longest subsequence that is present in both sequences in the same order, 
    consecutively.
    """
    m, n = len(X), len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i-1] == Y[j-1]:
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result

#################################################


################## FF Times #####################

def calculate_free_flow_times(od_paths_dict, network, show=False):
    """ Get ff time for all routes """
    length = pd.DataFrame(network.edges(data = True))
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


def _get_ff(path, network):
    """ Get ff time for a given route """
    length = pd.DataFrame(network.edges(data = True))
    time = length[2].astype('str').str.split(':',expand=True)[1]
    length[2] = time.str.replace('}','',regex=True).astype('float')
    rou=[]
    for i in range(len(path)):
        if i < len(path) - 1:
            for k in range(len(length[0])):
                if (path[i] == length[0][k]) and (path[i + 1] == length[1][k]):
                    rou.append(length[2][k])
    return sum(rou)

#################################################


################## Disk Ops #####################

def save_paths(routes, ff_times, paths_csv_save_path, routes_xml_save_path):
    """ Save paths and ff times to disk """
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



#################################################


#################### Main #######################

if __name__ == "__main__":
    params = get_params(kc.PARAMS_PATH)
    params = params[kc.PATH_GEN]

    number_of_paths = params[kc.NUMBER_OF_PATHS]
    beta = params[kc.BETA]
    weight = params[kc.WEIGHT]
    coeffs = params[kc.ROUTE_UTILITY_COEFFS]
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

    network = generate_network(connection_file_path, edge_file_path, route_file_path)
    check_od_integrity(network, origins, destinations)
    routes = create_routes(network, number_of_paths, origins, destinations, beta, weight, coeffs, num_samples, max_path_length)
    ff_times = calculate_free_flow_times(routes, network, show=True)
    save_paths(routes, ff_times, paths_csv_save_path, routes_xml_save_path)

#################################################