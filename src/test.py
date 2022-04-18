import src.utils as utils
import src.distributed_pca as dpca

# split dataset into n parts and safe to file as dataset0.csv...datasetn.csv

# spin up coordinator (server)


# loop on n_components,
#   invoke run(n_components) which returns P_hat
#   evaluate
#   write results to file

# 1 run is 1 full execution of algo 1
# We want to repeat runs with various t for given n and mal nodes
# We want to repeat runs with various n given t and mal nodes
# We want to repeat runs with various mal nodes given t and n

# Which question to ask:
# Given n nodes
# How big of a t do we need if we have m mal_nodes
# How many mal nodes can we tolerate if t is set

import sys
import socket 

def generate_nodes(n):
    """Given the number of nodes, generate hostanames, and ports for nodes
        Made to conviniently create nodes on the same machine
        n: number of nodes
        return: a list of tuples [(hostname, port)]
    """
    nodes = []
    hostname = hostname = socket.gethostbyname(socket.gethostname())
    starting_port = 50007
    
    for i in range(n):
        nodes.append((hostname, starting_port + i))

    return nodes 

def create_random_grid_graph(nodes, height, width=None, seed=1):
    """Creates a random grid graph from the nodes
        nodes: list of nodes in the graph
        height: height of the graph
        width: width of the graph, default to same as height
        seed: seed for randomization

        return: a dictionary {node : list of connected nodes}
    """
    pass

if __name__ == '__main__':
    if len(sys.argv) == 5:
        n_nodes = int(sys.argv[1])
        n_mal_nodes = int(sys.argv[2])
        mal_type = sys.argv[3]
        dataset_path = sys.argv[4]
        n_components = sys.argv[5]

    else:
        n_nodes = 10
        graph_height = 3
        n_mal_nodes = 0
        mal_type = 0
        dataset_path = 'iris_with_cluster.csv'
        n_components = 4
        

    #utils.split_dataset(dataset_path, n_components)

    nodes = generate_nodes(n_nodes)

    network_graph = create_random_grid_graph(nodes, graph_height)

    #create the actual nodes and start them?
    my_p2p_network = {}
    i = 0
    for node in network_graph.keys():
        
        # instantiate a new node 
        #start the node
        i += 1

    for node, connections in my_p2p_network:
        for conn in connections: 
            #add outbound connections
            pass
    
    # now the network is created and nodes are connected to each other

    # Part below can be put into a dist_pca func/class if needed
    for node in my_p2p_network.values():
        # Starts pca
        pass

    # Stop all nodes 

    """
    distPCA = dpca.DistributedPCA(
        n_nodes=n_nodes, n_mal_nodes=n_mal_nodes, mal_type=mal_type)
    P_hat = distPCA.fit_transform(
        dataset_path=dataset_path, n_components=n_components)
    """
    