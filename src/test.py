from os import abort
import sys
import socket
import time

import utils
import pca_p2p_node

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


def create_fully_connected_graph(nodes):
    """Creates a fully connected graph from the nodes
        nodes: list of nodes in the graph
        return: a dictionary {id: [(node), [(list of connected nodes)]]}
    """
    network_graph = {}
    for id in range(len(nodes)):
        network_graph[id] = [nodes[id], nodes[: id] + nodes[id + 1:]]

    return network_graph


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
        n_nodes = 3 #10
        graph_height = n_nodes
        n_mal_nodes = 0
        mal_type = 0
        dataset_path = 'iris_with_cluster.csv'
        n_components = 4

    #utils.split_dataset(dataset_path, n_components)

    nodes = generate_nodes(n_nodes)  # [(hostname, port)]
    print('Nodes')
    print(nodes)
    print()

    # {id: [(node), [(list of connected nodes)]]}
    network_graph = create_fully_connected_graph(nodes)
    #network_graph = create_random_grid_graph(nodes, graph_height)
    print('Network Graph')
    print(network_graph)
    print()

    # create the actual nodes and start them?
    my_p2p_network = {}  # dict {id : p2p_node object}
    for id in network_graph.keys():
        my_p2p_network[id] = pca_p2p_node.PCANode(network_graph[id][0][0], network_graph[id][0][1], id, debug=True)
    
    time.sleep(1)

    for node in my_p2p_network.values():
        node.start()

    print()
    time.sleep(1)

    for id, node in my_p2p_network.items():  # for each node
        # for each connection of that node
        for conn in network_graph[id][1]:
            node.connect_with(conn[0], conn[1])
            time.sleep(0.1)
            print()
            
        print()
        print('ended one round of connections')
        print()

    time.sleep(1)

    # now the network is created and nodes are connected to each other
    
    # Part below can be put into a dist_pca func/class if needed
    for node in my_p2p_network.values():
        #do pca
        pass

    #first non-pca test

    #test broadcast
    for node in my_p2p_network.values():
        node.broadcast(f'Hello from node {node.id}')
        time.sleep(0.1)

    #test disconnect
    #my_p2p_network[0].disconnect_with(my_p2p_network[1].id)

    #test send
    for from_id, from_node in my_p2p_network.items():
        for to_id in from_node.get_connected_nodes():
            from_node.send(to_id, f'Goodbye from {from_node.id}')
            time.sleep(0.1)
    #end

    # Stop all nodes
    time.sleep(3)
    print('stopping all')

    for node in my_p2p_network.values():  # for each node
        node.stop()
        time.sleep(1)
    
    time.sleep(15)

    #for id, node in my_p2p_network.items():
    #    print(f'[{id}] {node.get_connected_nodes()}')

    for node in my_p2p_network.values():  # for each node
        node.join()
        
    print('end test')
    
    """
    distPCA = dpca.DistributedPCA(
        n_nodes=n_nodes, n_mal_nodes=n_mal_nodes, mal_type=mal_type)
    P_hat = distPCA.fit_transform(
        dataset_path=dataset_path, n_components=n_components)
    """
