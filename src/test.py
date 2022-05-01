from cProfile import label
from cgi import test
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
from sklearn.cluster import KMeans
from sklearn import metrics

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
    def __init__(self, n_nodes, nodes=None, get_network_topology=None, node_class=pca_p2p_node.PCANode):
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
            self.network[id] = node_class(
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

    def make_malicious(self, ids, mal_type_class, attack_strategy):
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
            self.network[id].set_attack_strategy(attack_strategy)
            #print(type(self.network[id]))


class BaselineTest(): #original data centered around mean
    def __init__(self, n_mal_nodes=0, attack_type='None'):
        #self.n_components = None
        #self.projected_data = None
        self.total_variance = None
        self.rand_ind = None
        self.n_mal_nodes = n_mal_nodes
        self.attack_type = attack_type
        self.projected_data = None

    def do_k_means_benchmarking(self, data_mat, label_mat):
        n_true_clusters = np.unique(label_mat).size
        kmeans = KMeans(init='k-means++', n_clusters=n_true_clusters, random_state=0) 
        clustering = kmeans.fit_predict(data_mat)
        self.rand_ind = metrics.rand_score(label_mat, clustering)

    def run(self, data_mat, label_mat, n_components):
        self.n_components = n_components
        data_mat = np.array(data_mat)
        print(data_mat.shape)
        label_mat = np.array(label_mat)
        #self.label_mat = label_mat
        #self.projected_data = data_mat


        mean = np.mean(data_mat, axis=0)
        data_mat -= mean
        self.projected_data = data_mat
    
        U, D, E_T = np.linalg.svd(data_mat, full_matrices=True)
        explained_variance = (D**2) / (np.shape(data_mat)[0] - 1)
        self.total_variance = explained_variance.sum()
        self.do_k_means_benchmarking(data_mat, label_mat)

class PCATest(BaselineTest): #central PCA
    def __init__(self, n_mal_nodes=None, attack_type=None):
        super().__init__(n_mal_nodes, attack_type)

    def run(self, data_mat, label_mat, n_components, reduce_dim):
        data_mat = np.array(data_mat)
        print(data_mat.shape)
        label_mat = np.array(label_mat)
        self.n_components = n_components
        #self.label_mat = label_mat
        mean = np.mean(data_mat, axis=0)
        data_mat -= mean
        U, D, E_T = np.linalg.svd(data_mat, full_matrices=True)

        D_t = np.zeros((np.shape(data_mat)[0], self.n_components))
        for i in range(self.n_components):
            D_t[i][i] = D[i]

        E_t_T = E_T[:n_components]

        if not reduce_dim:
            projected_data = np.matmul(U, D_t)  # D_i_t
            projected_data = np.matmul(projected_data, E_t_T)
        else: 
            projected_data = np.matmul(data_mat, np.transpose(E_t_T))

    
        U, D, E_T = np.linalg.svd(projected_data, full_matrices=True)
        explained_variance = (D**2) / (np.shape(projected_data)[0] - 1)
        self.total_variance = explained_variance.sum()
        self.projected_data = projected_data
        self.do_k_means_benchmarking(projected_data, label_mat)

class DistPCATest(BaselineTest):
    def __init__(self, p2p_network, n_mal_nodes=None, attack_type=None):
        super().__init__()
        self.p2p_network = p2p_network.network
        self.n_mal_nodes = n_mal_nodes
        self.attack_type = attack_type

    def run(self, datasets, label_mat, n_components, reduce_dim):
        print()
        print('begin test')
        self.n_components = n_components
        #self.label_mat = label_mat
        for id, node in self.p2p_network.items():
            node.do_PCA(datasets[id], n_components, reduce_dim)

        time.sleep(5)

        while not self.p2p_network[0].pca_complete:
            time.sleep(5)

        projected_data = self.p2p_network[0].projected_global_data
        self.total_variance = self.p2p_network[0].total_variance

        # print(np.shape(dist_pca_projected_data))
        #plot(dist_pca_projected_data, dist_label_mat, 'distributed-network')
        print('end test')
        print()
        self.projected_data = projected_data
        self.do_k_means_benchmarking(projected_data, label_mat)

def plot_data(tests, label_mat, n_components, n_nodes, dataset_name):
    '''
    Input: gets a list of tests. Each test has the following:
    projected_data
    n_mal_nodes 
    attack_type

    TODO: create a figure with subplots. Each subplot should be the first two components of the projected data for that number of mal nodes
    There should be len(tests) no. of subplots
    save result to file with relevant text
    '''

    # print(np.shape(projected_data))
    
    fig, axs = plt.subplots(len(tests))
    #fig, axs = plt.subplots(3, 2)
    fig.suptitle(f'{dataset_name} dataset with {n_components} principal components, {n_nodes} nodes and {attack_type} attack')
    
    for i in range(len(tests)):
        t = tests[i]
        axs[i].scatter(t.projected_data[:,0], t.projected_data[:,1], c=label_mat)
        axs[i].set_title(f'{t.n_mal_nodes} malicious nodes')
        #plt.text(0, 0, s=f'proj data with principal components = {n_components}, \n no. of nodes = {n_nodes} \n {n_mal_nodes} of {attack_type} type')
    
    figname = f'dataset_name_{n_components}comps_{n_nodes}nodes_{attack_type}_attack_proj_data.png'
    fig.savefig(figname) 
    plt.show()

def plot_total_variance(x, pca_tests, n_components, n_nodes, dataset_name):
    """
    receives a dictionary of test objects. Each key is the attack type, value is list of test objects. Each test object has a field for total_variance
    """
    figname = dataset_name + '_' + str(n_nodes) + 'nodes_' + str(n_components) + 'comps_total_var.png'
    
    x_axis = np.arange(len(x))
    w = 0.2
    i = -1
    for attack_type, tests in pca_tests.items():
        total_vars = [test.total_variance for test in tests]
        plt.bar(x_axis+(i * w), total_vars, width=w, label=attack_type)
        i += 1

    #attack_type = 'reverse_order'
    #tests = pca_tests[attack_type]
    #total_vars = [test.total_variance for test in tests]
    #plt.bar(x_axis+(i * w), total_vars, width=w, label=attack_type)
    #i += 1

    plt.legend(pca_tests.keys())
    plt.xticks(x_axis,x)
    plt.title('Total variance as a function of malicious nodes')
    plt.text(len(x) - 3, total_vars[1], s=f'no. of principal components = {n_components}, \n no. of nodes = {n_nodes}')
    plt.xlabel('no. of malicious nodes')
    plt.ylabel('total variance explained by data (higher is better)')
    plt.savefig(figname)
    plt.show()


def plot_rand_ind(x, pca_tests, n_components, n_nodes, dataset_name):
    """
    receives a dictionary of test objects. Each key is the attack type, value is list of test objects. Each test object has a field for rand_ind
    """
    figname = dataset_name + '_' + str(n_nodes) + 'nodes_' + str(n_components) + 'comps_rand.png'
    
    x_axis = np.arange(len(x))
    w = 0.2
    i = -1
    for attack_type, tests in pca_tests.items():
        rand = [test.rand_ind for test in tests]

        plt.bar(x_axis+(i * w), rand, width=w, label=attack_type)
        i += 1
    
    plt.legend(pca_tests.keys())

    plt.xticks(x_axis,x)
    plt.title('Rand index for kmeans on projected data with m malicious nodes')
    plt.text(len(x) - 3, rand[1], s=f'no. of principal components = {n_components}, \n no. of nodes = {n_nodes}')
    plt.xlabel('no. of malicious nodes')
    plt.ylabel('Rand index for K-means (higher is better)')
    plt.savefig(figname)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 5:
        pass
    else:
        n_nodes = 3
        n_max_mal_nodes = n_nodes - 1
        #n_max_mal_nodes = 0
        #n_max_mal_nodes = 1
        dataset_path = '/datasets/iris/iris_with_cluster.csv'
        #dataset_path = '/datasets/cho/cho.csv'
        n_components = 2
        reduce_dim = False

    
    dataset_name = dataset_path.split('/')[-1]
    dataset_name = dataset_name[:-4]  # remove .csv

    dataset_path = os.getcwd() + dataset_path
    results_dir = os.getcwd() + '/results/' + dataset_name


    data_mat, label_mat = load_dataset(dataset_path, True)
    
    #attacks = {'randomize' : 1, 'reverse_order' : 2, 'make_perpendicular' : 3}
    #attacks = {'randomize' : 1}
    attacks = {'reverse_order' : 2}
    #attacks = {'make_perpendicular' : 3}

    pca_tests = {key : [] for key in attacks.keys()}
    print(pca_tests)

    baseline_test = BaselineTest()
    baseline_test.run(data_mat, label_mat, n_components)
    {tests.append(baseline_test) for tests in pca_tests.values()}

    central_pca_test = PCATest()
    central_pca_test.run(data_mat, label_mat, n_components, reduce_dim)
    #pca_tests.append(central_pca_test)
    {tests.append(central_pca_test) for tests in pca_tests.values()}

    # start tests on network
    local_datasets = split_dataset(data_mat, n_nodes)

    my_p2p_network = P2P_Network(n_nodes, node_class=pca_p2p_node.SecurePCA1)
    # now the network is created and nodes are connected to each other

    # test with no mal nodes
    pca_test = DistPCATest(my_p2p_network)
    #pca_tests.append(pca_test)
    {tests.append(pca_test) for tests in pca_tests.values()}
    pca_test.run(local_datasets, label_mat, n_components, reduce_dim)
    my_p2p_network.reset_all() 

    # iterate over increasing mal nodes
    
    malicious_node_ids = set()
    random.seed(0)
    for i in range(1, n_max_mal_nodes + 1):
        mal_id = random.randrange(1, n_nodes) # never make 0 malicious 
        while mal_id in malicious_node_ids:
            mal_id = random.randrange(1, n_nodes)
        malicious_node_ids.add(mal_id)
        print(f'made node {mal_id} malicious')

        for attack_type, attack_num in attacks.items():
            print(f'attack type {attack_type}, {attack_num}')
            
            my_p2p_network.make_malicious([mal_id], pca_p2p_node.MalPCANode, attack_num)

            pca_test = DistPCATest(my_p2p_network, i, attack_type)
            pca_test.run(local_datasets, label_mat, n_components, reduce_dim)

            pca_tests[attack_type].append(pca_test)
            my_p2p_network.reset_all() 

    # Stop all nodes
    #time.sleep(3)
    my_p2p_network.destroy()


    #print(pca_tests) 
    for attack_type, tests in pca_tests.items(): 
        plot_data(tests, label_mat, n_components, n_nodes, dataset_name)

    x = ['og data', 'c pca']
    x += [str(x) for x in range (n_max_mal_nodes + 1)]
    
    plot_total_variance(x, pca_tests, n_components, n_nodes, dataset_name)

    plot_rand_ind(x, pca_tests, n_components, n_nodes, dataset_name)
    
    
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
