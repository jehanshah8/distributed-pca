
import dis
from math import dist
import sys
import pandas as pd

from operator import matmul
import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip
from matplotlib import pyplot as plt
import os

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
            label_mats.append(chunks[i].iloc[:,-1])
        return data_mats, label_mats

def round_one(P_i, n_components):
        mean = np.mean(P_i, axis=0)
        P_i -= mean
        U, D, E_T = linalg.svd(P_i, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, E_T = svd_flip(U, E_T)
        #print(D)
        #print(D[:n_components])
    #
        #print()
    #
        #print(np.transpose(E_T))
        #print(np.transpose(E_T[:n_components]))
        return D[:n_components], np.transpose(E_T[:n_components])

def round_two(singular_values_recv, singular_vectors_recv, P_i, n_components):
    g_cov_mat = np.zeros((n_components, n_components))
    P_i = P_i.to_numpy()
    for i in range(len(singular_values_recv)):
        D_t = np.diag(singular_values_recv[i])
        #print(f'D_t shape = {np.shape(D_t)}')

        E_t = singular_vectors_recv[i]
        #print(f'E_t shape = {np.shape(E_t)}')

        P_i_t = np.matmul(P_i, E_t) 
        cov_mat = np.matmul(np.transpose(P_i_t), P_i_t)
        g_cov_mat = np.add(g_cov_mat, cov_mat) 
    
    eigen_values, eigen_vectors = np.linalg.eig(g_cov_mat)
    #print(f'eigen_values: {eigen_values}')
    #print(f'eigen_vectors: {eigen_vectors}')

    eigen_stuff = list(zip(eigen_values, eigen_vectors))
    eigen_stuff = sorted(eigen_stuff, key=lambda x: x[0], reverse=True)
    sorted_eigen_values, sorted_eigen_vectors = zip(*eigen_stuff)
    sorted_eigen_vectors = np.array(sorted_eigen_vectors)
    #print(f'eigen_vectors: {sorted_eigen_vectors}')

    #print()

    sorted_eigen_vectors = sorted_eigen_vectors[:, :n_components]
    #print(f't eigen_vectors: {sorted_eigen_vectors}')
    P_i_hat = np.matmul(P_i_t, sorted_eigen_vectors)
    P_i_hat = np.matmul(P_i_hat, np.transpose(sorted_eigen_vectors))
    return P_i_hat

        
    #np.concatenate(cov_mats, axis=0)
    #print(g_cov_mat)

def plot(lowDDataMat, labelMat, figname):
    '''
    Input:
        lowDDataMat: the 2-d data after PCA transformation obtained from pca function
        labelMat: the corresponding label of each observation obtained from loadData
    '''

    lowDDataMatT = np.transpose(lowDDataMat)
    #print(lowDDataMatT)
    plt.scatter(lowDDataMatT[0],lowDDataMatT[1], c=labelMat)
    plt.savefig(figname)
    #plt.show()

if __name__ == "__main__":
    
    n_nodes = 10
    dataset_path = '/datasets/iris/iris_with_cluster.csv'
    dataset_path = os.getcwd() + dataset_path
    n_components = 2

    ### central
    data_mat, label_mat = load_dataset(dataset_path)
    label_mat = np.array(label_mat)
    pca = PCA(2)  # project from 64 to 2 dimensions
    projected_2 = pca.fit_transform(data_mat)
    print(np.shape(projected_2))
    plot(projected_2, label_mat, 'central')
    


    local_data, local_labels = split_dataset(dataset_path, n_nodes, write=False)


    """
    local_data = []
    local_labels = []
    for i in range (n_nodes):
        X, Y = load_dataset(get_local_dataset_path(dataset_path, i))
        local_data.append(X)
        local_labels.append(Y)
    """

    singular_values_recv = []
    singular_vectors_recv = []

    for d in local_data:
        D_t, E_t = round_one(d, n_components)
        singular_values_recv.append(D_t)
        singular_vectors_recv.append(E_t)

    P_i_hats_recv = []
    for i in range(n_nodes):
        P_i_hats_recv.append(round_two(singular_values_recv, singular_vectors_recv, local_data[i], n_components))
    
    projected_1 = np.concatenate(P_i_hats_recv, axis=0)
    dist_label_mat = np.concatenate(local_labels, axis=0)
    dist_label_mat = np.array(dist_label_mat)

    plot(projected_1, dist_label_mat, 'distributed')
    print(np.shape(projected_1))
    #print()
    #print()
    #print()
    """
    projected_1 = np.matmul(X, np.transpose(Vt[:n_components]))

    """
    #print(label_mat)
    #print()
    #print(dist_label_mat)
    #
    #label_flag = True
#
    #for i in range(len(label_mat)):
    #    label_flag &= label_mat[i] == dist_label_mat[i]
#
    #print(label_flag)

    print('end test')