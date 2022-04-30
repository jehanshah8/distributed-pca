from functools import reduce
import sys
import socket
import time
import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from sklearn.decomposition import PCA

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

def create_random_grid_graph(nodes, height, width=None, seed=1):
    """Creates a random grid graph from the nodes
        nodes: list of nodes in the graph
        height: height of the graph
        width: width of the graph, default to same as height
        seed: seed for randomization
        return: a dictionary {node : list of connected nodes}
    """
    pass


def plot(projected_data, labels, figname):
    '''
    Input:
        lowDDataMat: the 2-d data after PCA transformation obtained from pca function
        labelMat: the corresponding label of each observation obtained from loadData
    '''
    # print(np.shape(projected_data))
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


def split_dataset(df, n_chunks):
    chunks = np.array_split(df, n_chunks)
    data_mats = []
    #label_mats = []
    for i in range(len(chunks)):
        data_mats.append(chunks[i].iloc[:, :])
        #data_mats.append(chunks[i].iloc[:, :-1])
        #label_mats.append(chunks[i].iloc[:, -1])
    return data_mats  # , label_mats


class P2P_Network():
    def __init__(self, n_nodes, nodes=None, get_network_topology=None):
        self.network_graph = None

        if nodes is None:
            self.nodes = self.generate_nodes(n_nodes)  # [(hostname, port)]
            print('Nodes')
            print(self.nodes)
            print()
        else:
            self.nodes = nodes

        if get_network_topology is None:
            get_network_topology = self.create_fully_connected_graph

        # {id: [(node), [(list of connected nodes)]]}
        self.network_graph = get_network_topology(self.nodes)
        #network_graph = create_random_grid_graph(nodes, graph_height)
        print('Network Graph')
        print(self.network_graph)
        print()

        # create the actual nodes and start them?
        self.network = {}  # dict {id : p2p_node object}
        for id in self.network_graph.keys():
            self.network[id] = pca_p2p_node.PCANode(
                self.network_graph[id][0][0], self.network_graph[id][0][1], id, debug=True)

        time.sleep(1)

        for node in self.network.values():
            node.start()

        print()
        time.sleep(1)

        for id, node in self.network.items():  # for each node
            # for each connection of that node
            for conn in self.network_graph[id][1]:
                node.connect_with(conn[0], conn[1])
                time.sleep(0.1)
                print()

            print()
            print('ended one round of connections')
            print()

        time.sleep(1)

        print('network created')

    @staticmethod
    def generate_nodes(n_nodes):
        """Given the number of nodes, generate hostanames, and ports for nodes
            Made to conviniently create nodes on the same machine
            n_nodes: number of nodes
            return: a list of tuples [(hostname, port)]
        """
        nodes = []
        hostname = hostname = socket.gethostbyname(socket.gethostname())
        starting_port = 50007

        for i in range(n_nodes):
            nodes.append((hostname, starting_port + i))

        return nodes

    @staticmethod
    def create_fully_connected_graph(nodes):
        """Creates a fully connected graph from the nodes
            nodes: list of nodes in the graph
            return: a dictionary {id: [(node), [(list of connected nodes)]]}
        """
        network_graph = {}
        for id in range(len(nodes)):
            network_graph[id] = [nodes[id], nodes[: id] + nodes[id + 1:]]

        return network_graph

    def reset_all(self):
        for node in self.network.values():  # for each node
            node.reset()

    def destroy(self):
        print('stopping all')
        for node in self.network.values():  # for each node
            node.stop()
            time.sleep(0.1)

        time.sleep(1)

        # for id, node in network.items():
        #    print(f'[{id}] {node.get_connected_nodes()}')

        for node in self.network.values():  # for each node
            node.join()

    def make_malicious(self, ids, mal_type_class):
        """Takes in a list of ids of the nodes to make malicious
        """
        #for id in ids:
        #    node = self.network[id]
        #    node_connections = node.get_connected_nodes()
        #    node.stop()
        #    time.sleep(0.1)
        #    node.join()
#
        #    #time.sleep(1)
#
        #    mal_node = mal_type_class(
        #        self.network_graph[id][0][0], self.network_graph[id][0][1], id, debug=True)
#
        #    #time.sleep(1)
#
        #    mal_node.start()
        #    time.sleep(1)
        #    print()
#
        #    for conn_id in node_connections:
        #        # {id: [(node), [(list of connected nodes)]]}
        #        mal_node.connect_with(
        #            self.network_graph[int(conn_id)][0][0], self.network_graph[int(conn_id)][0][1])
        #        time.sleep(0.1)
        #        print()
#
        #    self.network[id] = mal_node
        for id in ids:
            self.network[id].__class__ = mal_type_class 
            #print(type(self.network[id]))


class BaselineTest(): #original data centered around mean
    def __init__(self):
        self.n_components = None
        self.projected_data = None
        self.total_variance = None

    def run(self, data_mat, n_components, results_dir):
        self.n_components = n_components
        self.projected_data = data_mat

        mean = np.mean(data_mat, axis=0)
        data_mat -= mean
        self.projected_data = data_mat

        U, D, E_T = np.linalg.svd(data_mat, full_matrices=True)
        explained_variance = (D**2) / (np.shape(data_mat)[0] - 1)
        self.total_variance = explained_variance.sum()

class PCATest(): #central PCA
    def __init__(self):
        self.n_components = None
        self.projected_data = None
        self.total_variance = None

    def run(self, data_mat, n_components, results_dir, reduce_dim):
        self.n_components = n_components
        mean = np.mean(data_mat, axis=0)
        data_mat -= mean
        U, D, E_T = np.linalg.svd(data_mat, full_matrices=True)

        D_t = np.zeros((np.shape(data_mat)[0], self.n_components))
        for i in range(self.n_components):
            D_t[i][i] = D[i]

        E_t_T = E_T[:n_components]

        if not reduce_dim:
            self.projected_data = np.matmul(U, D_t)  # D_i_t
            self.projected_data = np.matmul(self.projected_data, E_t_T)
        else: 
            self.projected_data = np.matmul(data_mat, np.transpose(E_t_T))

    
        U, D, E_T = np.linalg.svd(self.projected_data, full_matrices=True)
        explained_variance = (D**2) / (np.shape(self.projected_data)[0] - 1)
        self.total_variance = explained_variance.sum()


class DistPCATest():
    def __init__(self, p2p_network):
        self.p2p_network = p2p_network.network
        self.n_components = None
        self.projected_data = None
        self.total_variance = None

    def run(self, datasets, n_components, results_dir, reduce_dim):
        print()
        print('begin test')
        self.n_components = n_components
        for id, node in self.p2p_network.items():
            node.do_PCA(datasets[id], n_components, reduce_dim)

        time.sleep(5)

        while not self.p2p_network[0].pca_complete:
            time.sleep(5)

        self.projected_data = self.p2p_network[0].projected_global_data
        self.total_variance = self.p2p_network[0].total_variance

        # print(np.shape(dist_pca_projected_data))
        #plot(dist_pca_projected_data, dist_label_mat, 'distributed-network')
        print('end test')
        print()


if __name__ == '__main__':
    if len(sys.argv) == 5:
        pass
    else:
        n_nodes = 3
        n_max_mal_nodes = n_nodes - 1
        #n_max_mal_nodes = 0
        dataset_path = '/datasets/iris/iris_with_cluster.csv'
        #dataset_path = '/datasets/cho/cho.csv'
        n_components = 2
        reduce_dim = True

    
    dataset_name = dataset_path.split('/')[-1]
    dataset_name = dataset_name[:-4]  # remove .csv

    dataset_path = os.getcwd() + dataset_path
    results_dir = os.getcwd() + '/results/' + dataset_name


    data_mat, label_mat = load_dataset(dataset_path, True)

    pca_tests = []

    baseline_test = BaselineTest()
    baseline_test.run(data_mat, n_components, results_dir)
    pca_tests.append(baseline_test)

    central_pca_test = PCATest()
    central_pca_test.run(data_mat, n_components, results_dir, reduce_dim)
    pca_tests.append(central_pca_test)

    # start tests on network
    local_datasets = split_dataset(data_mat, n_nodes)

    my_p2p_network = P2P_Network(n_nodes)
    # now the network is created and nodes are connected to each other

    # test with no mal nodes
    pca_test = DistPCATest(my_p2p_network)
    pca_tests.append(pca_test)
    pca_test.run(local_datasets, n_components, results_dir, reduce_dim)
    my_p2p_network.reset_all() 

    # iterate over increasing mal nodes
    malicious_node_ids = set()
    random.seed(0)
    for i in range(n_max_mal_nodes):
        mal_id = random.randrange(1, n_nodes) # never make 0 malicious 
        while mal_id in malicious_node_ids:
            mal_id = random.randrange(1, n_nodes)
        malicious_node_ids.add(mal_id)
        my_p2p_network.make_malicious([mal_id], pca_p2p_node.MalPCANode1)

        pca_test = DistPCATest(my_p2p_network)
        pca_tests.append(pca_test)
        pca_test.run(local_datasets, n_components, results_dir, reduce_dim)

        my_p2p_network.reset_all() 

    # Stop all nodes
    #time.sleep(3)
    my_p2p_network.destroy()

    x = ['original data', 'central pca']
    x += [str(x) + 'mal nodes' for x in range (n_max_mal_nodes + 1)]
    total_vars = [test.total_variance for test in pca_tests]
    plt.plot(x, total_vars)
    plt.legend()
    plt.show()

    print('finished')





















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
