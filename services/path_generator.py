import networkx as nx

from human_learning import logit

def path_generator(G, origin, destination, weight, avoid_nodes, beta):     # Maybe termination limit?
    current_node = origin
    path = [origin]
    distance_to_destination = nx.single_source_dijkstra_path_length(G, destination, weight = weight)  # was origin

    while current_node != destination:
        
        reached_to_destination = False
        all_neighbors = list(G.neighbors(current_node))

        options = list()
        for node in all_neighbors:
            if (node == destination) or (list(G.neighbors(node)) and (not node in path)):     # if node is destination or a non-visited non-deadend
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

        if reached_to_destination: break

        chosen_index = logit(beta, costs)
        chosen_node = options[chosen_index]

        path.append(chosen_node)
        current_node = chosen_node
    
    return path