import networkx as nx

from human_learning import logit


def path_generator(network, origin, destination, weight, avoid_nodes, beta):

    path = list()   # our path
    visited_nodes_abs_names = set()     # visited node memory

    distance_to_destination = nx.single_source_dijkstra_path_length(network, destination, weight = weight)    # heuristic
    
    reached_to_destination = False
    current_node = origin   # we start from the origin

    while True:
        # Add current node to path and visited log
        path.append(current_node)
        visited_nodes_abs_names.add(abs_node_id(current_node))
        
        # Find reachable nodes
        all_neighbors = list(network.neighbors(current_node))

        # Find reacheble AND feasible nodes
        options = list()
        for node in all_neighbors:
            if (node == destination) or (list(network.neighbors(node)) and (not abs_node_id(node) in visited_nodes_abs_names)):     
                # if node is (destination) or (non-visited, non-deadend)
                options.append(node)

        # if we see no feasible node
        if not options: return path_generator(network, origin, destination, weight, avoid_nodes, beta)   # Restart

        # For each node, find how likely it should be to pick it
        costs = list()
        for node in options:
            if node == destination:     # But if we see we found the destination
                path.append(node)
                reached_to_destination = True
                break
            elif node in avoid_nodes:   # Please think about this!
                costs.append(distance_to_destination[node] * 5)  # if we saw this node in any previous path, discourage
            else:
                costs.append(distance_to_destination[node])

        if reached_to_destination:
            break   # Path is ready, connects origin to dest
        else:
            chosen_index = logit(beta, costs)    # Pick the next node
            current_node = options[chosen_index]
    
    return path




def cursed_path_generator(network, origin, destination, weight, avoid_nodes, beta):
    
    path = list()   # Our path
    visited_nodes_abs_names = set()     # Visited node memory
    cursed_nodes = set()    # Nodes that LEAD TO deadend

    distance_to_destination = nx.single_source_dijkstra_path_length(network, destination, weight = weight)    # Our heuristic
    
    reached_to_destination = False
    current_node = origin   # We start from the origin

    while True:
        # Add current node to path and visited
        path.append(current_node)
        visited_nodes_abs_names.add(abs_node_id(current_node))
        
        # Find reachable nodes
        all_neighbors = list(network.neighbors(current_node))

        # Find reacheble AND feasible nodes
        options = list()
        for node in all_neighbors:
            if (node == destination) or (list(network.neighbors(node)) and (not abs_node_id(node) in visited_nodes_abs_names) and (node not in cursed_nodes)):     
                # if node is (destination) or (non-deadend, non-visited, non-cursed)
                options.append(node)

        # if we see no feasible node
        if not options: 
            if (path[-1] != origin):
                cursed_nodes.add(path.pop())    # Remember this node leads to nothing but deadend
            current_node = path.pop()
            continue   # Take a step back, try again

        # For each node, find how likely it should be to pick it
        costs = list()
        for node in options:
            if node == destination:     # But if we see we found the destination
                path.append(node)
                reached_to_destination = True
                break
            elif node in avoid_nodes:   # Please think about this!
                costs.append(distance_to_destination[node] * 3)  # if we saw this node in any previous path, discourage
            else:
                costs.append(distance_to_destination[node])

        if reached_to_destination:
            break   # Path is ready, connects origin to dest
        else:
            chosen_index = logit(beta, costs)
            current_node = options[chosen_index]    # Pick the next node
    
    return path




def abs_node_id(node_id):
    return node_id.removeprefix("-")