from queue import Full
import sys
import pandas as pd

from operator import matmul
import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip
from matplotlib import pyplot as plt
import os

def plot(projected_data, labels, figname):
    '''
    Input:
        lowDDataMat: the 2-d data after PCA transformation obtained from pca function
        labelMat: the corresponding label of each observation obtained from loadData
    '''
    print(np.shape(projected_data))
    projected_data = np.transpose(projected_data)
    #print(lowDDataMatT)
    plt.scatter(projected_data[0],projected_data[1], c=labels)
    plt.savefig(figname)

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
        label_mats = []
        for i in range(len(chunks)):
            data_mats.append(chunks[i].iloc[:, :-1])
            label_mats.append(chunks[i].iloc[:,-1])
        return data_mats, label_mats

def round_one(P_i, n_components):
        mean = np.mean(P_i, axis=0)
        P_i -= mean
        #print(f'data shape = {np.shape(local_datasets[i])}')
        U_i, D_i, E_i_T = linalg.svd(P_i, full_matrices=True)
        #print(f'U shape = {np.shape(U_i)}')
        #print(f'D shape = {np.shape(D_i)}')
        #print(f'E_T shape = {np.shape(E_i_T)}')
        # flip eigenvectors' sign to enforce deterministic output
        #U_i, E_i_T = svd_flip(U_i, E_i_T)
        
        # my way
        P_i_t = matmul(P_i, np.transpose(E_i_T[:n_components]))
        P_i_t = matmul(P_i_t, E_i_T)
        #P_i_t = P_i.to_numpy()

        print(f'my way{P_i_t}')
        #print(f'data shape after t = {np.shape(local_datasets[i])}')

        return D_i[:n_components], np.transpose(E_i_T[:n_components]), P_i_t

def round_two(singular_values_recv, singular_vectors_recv, P_i_t, n_components):
    #P_i_t = P_i_t.to_numpy()
    #print(f'shape of local data {np.shape(P_i_t)}')

    g_cov_mat = np.zeros((np.shape(P_i_t)[1], np.shape(P_i_t)[1]))  #either txt or dxd? 
    #print(f'shape of g_cov_mat {np.shape(g_cov_mat)}')

    #for data from each node
    for i in range(len(singular_values_recv)):
        
        D_i_t = np.zeros((np.shape(P_i_t)[0], n_components))
        for j in range(n_components):
            D_i_t[j][j] = singular_values_recv[i][j]

        #print(f'D_t shape = {np.shape(d_i_t)}')

        E_i_t = singular_vectors_recv[i] 
        #print(f'E_t shape = {np.shape(e_i_t)}')
        

        #One way to get cov mats
        
        cov_mat = np.matmul(E_i_t, np.transpose(D_i_t))
        cov_mat = np.matmul(cov_mat, D_i_t)
        cov_mat = np.matmul(cov_mat, np.transpose(E_i_t))
        
        #second way to get cov mats
        """
        D_t = np.diag(singular_values_recv[i])
        #print(f'D_t shape = {np.shape(D_t)}')

        E_t = singular_vectors_recv[i]
        #print(f'E_t shape = {np.shape(E_t)}')
                            #p = n x d
        
        P_i_t = np.matmul(P, E_t) #E_t = d x t # projection of 
        # local data on t components using its own singular vectors
        
        cov_mat = np.matmul(np.transpose(P_i_t), P_i_t)
        g_cov_mat = np.add(g_cov_mat, cov_mat) 
        # E = d x d -> first t columns 
        #E_t * (D_t)transpose = d x t * t x n
        #(E_t * (D_t)transpose) * D_t = d x n * n x t = d x t 
        #d x t * t x d = d x d 
        #should be t x t 
        """


        g_cov_mat = np.add(g_cov_mat, cov_mat) 


    # CERTAIN ABOUT BELOW 

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
    #P_i_hat = np.matmul(P_i_hat, np.transpose(sorted_eigen_vectors)) ## adding this means no dim reduction on transform
    return P_i_hat

if __name__ == "__main__":
    
    n_nodes = 5
    #dataset_path = '/datasets/cho/cho.csv'
    dataset_path = '/datasets/iris/iris_with_cluster.csv'
    dataset_path = os.getcwd() + dataset_path
    n_components = 2

    
    
    data_mat, label_mat = load_dataset(dataset_path)
    label_mat = np.array(label_mat)

    ### og data
    #plot(data_mat., label_mat, 'og-data')

    #print(np.shape(data_mat))
    
    ### central
    pca = PCA(n_components) 
    projected_2 = pca.fit_transform(data_mat)
    #print(np.shape(projected_2))
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
    P_i_t_recv = []
    for i in range(n_nodes):
        D_t, E_t, P_t = round_one(local_data[i], n_components)
        singular_values_recv.append(D_t)
        singular_vectors_recv.append(E_t)
        P_i_t_recv.append(P_t)

    print('end round 1')

    P_i_hats_recv = []
    for i in range(n_nodes):
        P_hat = round_two(singular_values_recv, singular_vectors_recv, P_i_t_recv[i], n_components)
        P_i_hats_recv.append(P_hat)
    print('end round 2')

    projected_1 = np.concatenate(P_i_hats_recv, axis=0)
    dist_label_mat = np.concatenate(local_labels, axis=0)
    dist_label_mat = np.array(dist_label_mat)

    plot(projected_1, dist_label_mat, 'distributed')
    #print(np.shape(projected_1))
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