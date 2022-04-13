import sys
import numpy as np
import pandas as pd
import src.server as server
import src.distributedPCA as distributedPCA

def load_dataset(filename = 'iris_with_cluster.csv'):
    """
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_array=line.strip().split(',')
        records = []
        for attr in line_array[:-1]:
            records.append(float(attr))
        data_mat.append(records)
        label_mat.append(int(line_array[-1]))
    data_mat = np.array(data_mat)
    
    label_mat = np.array(label_mat)
    
    return data_mat,label_mat
    """

    df = pd.read_csv(filename)
    data_mat = df.iloc[:,:-1]
    label_mat = df.iloc[:,-1]
    return data_mat, label_mat


def split_dataset(X, Y=None, n_nodes=2): 
        assert n_nodes > 0, "Invalid number of nodes"

        if (X != None): 
            local_featuresets = np.array_split(X, n_nodes)
        else: 
            local_featuresets = None

        if (Y != None): 
            local_featuresets = np.array_split(Y, n_nodes)
        else: 
            local_labelsets = None

        return local_featuresets, local_labelsets

if __name__ == '__main__':
    if len(sys.argv) == 5:
        filename = sys.argv[1]
        n_nodes = int(sys.argv[2])
        n_components = int(sys.argv[3])
        n_mal_nodes = int(sys.argv[4])
    else:
        filename = 'iris_with_cluster.csv'
        n_nodes = 1
        n_components = 5
        n_mal_nodes = 0

    data_mat, label_mat = load_dataset(filename)
    #print(data_mat)
    
    local_featuresets, local_labelsets = split_dataset(data_mat, label_mat, n_nodes)

    nodes = []
    for i in range(n_nodes - n_mal_nodes):
        nodes.append(distributedPCA.LocalPCA(n_components=n_components))

    for i in range(n_mal_nodes): 
        nodes.append(distributedPCA.MalLocalPCA(n_components=n_components))

    # Assuming one central coordinator but each node executing global pca 
    for i in range(n_nodes):
        nodes[i].fit(local_featuresets[i])

    globalPCA = GlobalPCA(n_components=n_components, nodes=local_nodes)
    globalPCA.fit()
    
# python3 test.py iris_with_cluster.csv 10 5 0
    