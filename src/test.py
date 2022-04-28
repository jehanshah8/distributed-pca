from multiprocessing.connection import wait
import sys
import socket
import time
import json
import os 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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

def plot(projected_data, labels, figname):
    '''
    Input:
        lowDDataMat: the 2-d data after PCA transformation obtained from pca function
        labelMat: the corresponding label of each observation obtained from loadData
    '''
    #print(np.shape(projected_data))
    projected_data = np.transpose(projected_data)
    # print(lowDDataMatT)
    plt.scatter(projected_data[0], projected_data[1], c=labels)
    plt.savefig(figname)
    plt.show()

def load_dataset(dataset_path='iris_with_cluster.csv', seperate_feats=True):
    df = pd.read_csv(dataset_path)
    if seperate_feats:
        data_mat = df.iloc[:, :-1]
        label_mat = df.iloc[:, -1]
        return data_mat, label_mat
    else:
        return df


def get_local_dataset_path(dataset_path, id):
    local_dataset_path = dataset_path[:-4] + str(id) + ".csv"
    return local_dataset_path


def split_dataset(dataset_path, n_chunks, write=True):
    df = load_dataset(dataset_path, False)
    chunks = np.array_split(df, n_chunks)

    if write:
        for i in range(len(chunks)):
            chunks[i].to_csv(get_local_dataset_path(
                dataset_path, i), index=False)
    else:
        data_mats = []
        #chunks[:, :-1]
        label_mats = []
        #chunks[:, -1]
        for i in range(len(chunks)):
            data_mats.append(chunks[i].iloc[:, :-1])
            label_mats.append(chunks[i].iloc[:, -1])
        return data_mats, label_mats


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


def create_PCA_p2p_network(n_nodes, n_mal_nodes, get_network_topology=create_fully_connected_graph):
    nodes = generate_nodes(n_nodes)  # [(hostname, port)]
    print('Nodes')
    print(nodes)
    print()

    # {id: [(node), [(list of connected nodes)]]}
    network_graph = get_network_topology(nodes)
    #network_graph = create_random_grid_graph(nodes, graph_height)
    print('Network Graph')
    print(network_graph)
    print()

    # create the actual nodes and start them?
    p2p_network = {}  # dict {id : p2p_node object}
    for id in network_graph.keys():
        p2p_network[id] = pca_p2p_node.PCANode(
            network_graph[id][0][0], network_graph[id][0][1], id, debug=True)

    time.sleep(1)

    for node in p2p_network.values():
        node.start()

    print()
    time.sleep(1)

    for id, node in p2p_network.items():  # for each node
        # for each connection of that node
        for conn in network_graph[id][1]:
            node.connect_with(conn[0], conn[1])
            time.sleep(0.1)
            print()

        print()
        print('ended one round of connections')
        print()

    time.sleep(1)

    return p2p_network


def stop_p2p_network(p2p_network):
    print('stopping all')
    for node in p2p_network.values():  # for each node
        node.stop()
        time.sleep(1)

    time.sleep(15)

    # for id, node in p2p_network.items():
    #    print(f'[{id}] {node.get_connected_nodes()}')

    for node in p2p_network.values():  # for each node
        node.join()

def run_pca(p2p_network, local_datasets, dist_label_mat, n_components, results_dir):
    
    for id, node in my_p2p_network.items():
        node.do_PCA(local_datasets[id], n_components)

    time.sleep(10)

    while not my_p2p_network[0].pca_complete:
        time.sleep(5)

    projected_data = my_p2p_network[0].projected_global_data
    total_var = my_p2p_network[0].total_variance
    
    #print(np.shape(dist_pca_projected_data))
    #plot(dist_pca_projected_data, dist_label_mat, 'distributed-network')


if __name__ == '__main__':
    if len(sys.argv) == 5:
        pass
    else:
        n_nodes = 3 
        n_mal_nodes = 0
        dataset_path = '/datasets/iris/iris_with_cluster.csv'
        #dataset_path = '/datasets/cho/cho.csv'
        n_components = 2

    print('begin test')
    dataset_name = dataset_path.split('/')[-1]
    dataset_name = dataset_name[:-4] #remove .csv

    dataset_path = os.getcwd() + dataset_path 
    results_dir = os.getcwd() + '/results/' + dataset_name

    my_p2p_network = create_PCA_p2p_network(n_nodes, n_mal_nodes, create_fully_connected_graph)
    print('network created')
    # now the network is created and nodes are connected to each other

    local_datasets, local_labelsets = split_dataset(dataset_path, n_nodes, write=False)
    dist_label_mat = np.concatenate(local_labelsets, axis=0)
    dist_label_mat = np.array(dist_label_mat)

    run_pca(my_p2p_network, local_datasets, dist_label_mat, n_components, results_dir)

    

    ## Stop all nodes
    time.sleep(3)
    stop_p2p_network(my_p2p_network)

    print('end test')
















    # first non-pca test

    # test broadcast
    # for node in my_p2p_network.values():
    #    node.broadcast(f'Hello from node {node.id}')
    #    time.sleep(0.1)

    # test disconnect
    # my_p2p_network[0].disconnect_with(my_p2p_network[1].id)

    # test send
    # for from_id, from_node in my_p2p_network.items():
    #    for to_id in from_node.get_connected_nodes():
    #        from_node.send(to_id, f'Goodbye from {from_node.id}')
    #        time.sleep(0.1)
    # end

    #singular_values = np.eye(10)
    #msg = {}
    #msg['singular_values'] = np.ndarray.tolist(singular_values)
    # msg['singular_vectors'] =
    #my_p2p_network[0].send(my_p2p_network[1].id, msg)
