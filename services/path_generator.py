import networkx as nx

from human_learning import logit

def path_generator(G, origin, destination, weight, avoid_nodes, beta):
    current_node = origin
    path = [origin]
    visited_nodes_abs_names = set()
    visited_nodes_abs_names.add(node_to_abs_id(origin))
    distance_to_destination = nx.single_source_dijkstra_path_length(G, destination, weight = weight)  # was origin
    reached_to_destination = False

    while True:
        
        all_neighbors = list(G.neighbors(current_node))

        options = list()
        for node in all_neighbors:
            if (node == destination) or (list(G.neighbors(node)) and (not node in path) and (not node_to_abs_id(node) in visited_nodes_abs_names)):     # if node is destination or a non-visited non-deadend
                options.append(node)

        if not options: return path_generator(G, origin, destination, weight, avoid_nodes, beta)   # if we filtered out all possible nodes, restart

        costs = list()
        for node in options:
            if node == destination:
                path.append(node)
                reached_to_destination = True
                break
            elif node in avoid_nodes:   # Please think about this!
                costs.append(distance_to_destination[node] * 3)  # if we saw this node in any previous path, discourage
            else:
                costs.append(distance_to_destination[node])

        if reached_to_destination:
            break
        else:
            chosen_index = logit(beta, costs)
            chosen_node = options[chosen_index]

            path.append(chosen_node)
            visited_nodes_abs_names.add(node_to_abs_id(chosen_node))
            current_node = chosen_node
    
    return path



def cursed_path_generator(G, origin, destination, weight, avoid_nodes, beta):
    current_node = origin
    path = [origin]
    cursed_nodes = set()
    distance_to_destination = nx.single_source_dijkstra_path_length(G, destination, weight = weight)  # was origin
    reached_to_destination = False

    while True:
        
        all_neighbors = list(G.neighbors(current_node))

        options = list()
        for node in all_neighbors:
            if ((node == destination) or (list(G.neighbors(node)) and (not node in path))) and (node not in cursed_nodes):     # if node is destination or a non-visited non-deadend
                options.append(node)

        if not options: 
            if (path[-1] != origin):
                cursed_nodes.add(path.pop())
            current_node = path[-1]
            continue   # if we filtered out all possible nodes, take a step back

        costs = list()
        for node in options:
            if node == destination:
                path.append(node)
                reached_to_destination = True
                break
            elif node in avoid_nodes:   # Please think about this!
                costs.append(distance_to_destination[node] * 3)  # if we saw this node in any previous path, discourage
            else:
                costs.append(distance_to_destination[node])

        if reached_to_destination:
            break
        else:
            chosen_index = logit(beta, costs)
            chosen_node = options[chosen_index]

            path.append(chosen_node)
            current_node = chosen_node
    
    return path


def node_to_abs_id(node_id):
    if node_id[0]=="-":
        print("Turned node_id from %s to %s" % (node_id, node_id[1:]))
        node_id = node_id[1:]
    return node_id