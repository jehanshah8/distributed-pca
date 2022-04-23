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
    #print(np.shape(projected_data))
    projected_data = np.transpose(projected_data)
    # print(lowDDataMatT)
    plt.scatter(projected_data[0], projected_data[1], c=labels, edgecolor='none', alpha=1,
                cmap=plt.cm.get_cmap('spring', 10))
    # plt.savefig(figname)
    plt.colorbar()
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

    ### line 6 equivalent
    D_i_t = np.zeros((np.shape(P_i)[0], n_components))
    for i in range(n_components):
        D_i_t[i][i] = D_i[i]

    E_i_t_T = E_i_T[:n_components]
    
    #algo way
    P_i_t = matmul(U_i, D_i_t)  # D_i_t
    P_i_t = matmul(P_i_t, E_i_t_T)

    # new way
    #P_i_t = P_i.to_numpy()

    #print(f'round two way')
    #print(f'{P_i_t}')
    #print(f'{np.shape(P_i_t)}')
    ###

    return D_i[:n_components], np.transpose(E_i_T[:n_components]), P_i_t


def round_two(singular_values_recv, singular_vectors_recv, P_i_t, n_components):
    g_cov_mat = np.zeros((np.shape(P_i_t)[1], np.shape(P_i_t)[1]))  # dxd
    #print(f'shape of g_cov_mat {np.shape(g_cov_mat)}')

    # for data from each node
    for i in range(len(singular_values_recv)):

        D_i_t = np.zeros((np.shape(P_i_t)[0], n_components))
        #D_i_t = np.zeros_like(P_i_t)
        for j in range(n_components):
            D_i_t[j][j] = singular_values_recv[i][j]
        #print(f'D_t shape = {np.shape(d_i_t)}')

        E_i_t = singular_vectors_recv[i]
        #print(f'E_t shape = {np.shape(e_i_t)}')

        cov_mat = np.matmul(E_i_t, np.transpose(D_i_t))
        cov_mat = np.matmul(cov_mat, D_i_t)
        cov_mat = np.matmul(cov_mat, np.transpose(E_i_t))
        cov_mat = cov_mat 
        g_cov_mat = np.add(g_cov_mat, cov_mat)

    g_cov_mat = g_cov_mat / (len(singular_values_recv) * np.shape(P_i_t)[0])

    #print(g_cov_mat)
    # CERTAIN ABOUT BELOW
    U, D, E_T = linalg.svd(g_cov_mat, full_matrices=True)
    
    P_i_hat = np.matmul(P_i_t, np.transpose(E_T[:n_components]))
    # adding below means no dim reduction on transform
    P_i_hat = np.matmul(P_i_hat, E_T[:n_components])
    print(P_i_hat.shape)
    return P_i_hat, g_cov_mat

def TestEvaluationMetric2(P_t, P_t_hat, n_components):
    """ Jaccard Distance:
        P_t : (n x t)
        P_t_hat : (n x t)
    """
    Jaccard_Distance = []
    
    for i in range(0,n_components): # for each reduced dimensions
        sum_max = 0
        sum_min = 0
        for j in range(0,P_t.shape[0]): # for each sample 
            sum_max += max( P_t[j,i], P_t_hat[j,i] )
            sum_min += min( P_t[j,i], P_t_hat[j,i] )
            # print(sum_max, sum_min)
        # print(P_t, P_t_hat)
        # print(sum_min, sum_max)
        Jaccard_Distance.append( (1 - abs(sum_min/sum_max) ) )
    return Jaccard_Distance

def TestEvaluationMetric(X, P_t_hat, n_components):
    subtraction = ((X)**2 - (P_t_hat)**2)
    
def TestEvaluationMetric3(C, D, n_components):
    Uc, Dc, E_Tc = linalg.svd(C, full_matrices=True)
    Ud, Dd, E_Td = linalg.svd(D, full_matrices=True)
    # print(Dc)
    # print(Dd)
    Dc = (Dc**2) / (C.shape[0] - 1)
    Dd = (Dd**2) / (D.shape[0] - 1)
    Percent_Exp_Variance_Central = sum(Dc[0:n_components])#/sum(Dc) * 100
    Percent_Exp_Variance_Dist = sum(Dd[0:n_components])#/sum(Dc) * 100 # divide by sum of Dc not Dd
    # print(Percent_Exp_Variance_Dist/Percent_Exp_Variance_Central)
    # plt.bar(range(len(Dc)),Dc)
    # plt.show()
    # plt.bar(range(len(Dd)),Dd)
    # plt.show()
    return Percent_Exp_Variance_Dist#/Percent_Exp_Variance_Central
    

if __name__ == "__main__":

    
    n_nodes_all = 20
    dataset_path = 'C:/Users/saimu/OneDrive - purdue.edu/Purdue Graduate School/MS_Electrical_Computer_Engineering/Second Year 2020-2021/ECE 60872 Fault Tolerance Design/Project/distributed-pca-p2p-network/distributed-pca-p2p-network/datasets/cho/cho.csv'
    # dataset_path = 'C:/Users/saimu/OneDrive - purdue.edu/Purdue Graduate School/MS_Electrical_Computer_Engineering/Second Year 2020-2021/ECE 60872 Fault Tolerance Design/Project/distributed-pca-p2p-network/distributed-pca-p2p-network/datasets/iris/iris_with_cluster.csv'
    # dataset_path = os.getcwd() + dataset_path
    n_components_all = 16

    data_mat, label_mat = load_dataset(dataset_path)
    label_mat = np.array(label_mat)
    
    #mean = np.mean(data_mat, axis=0)
    #data_mat -= mean
    #data_mat = data_mat.to_numpy()
    #cov_mat = np.matmul(np.transpose(data_mat), data_mat)
    
    #cov_mat = np.cov(np.transpose(data_mat))
    #print(cov_mat)

    ratio_dict = {}
    for n_nodes in range(1, n_nodes_all):
        list_temp = []
        for n_components in range(2, n_components_all):
            local_data, local_labels = split_dataset(
                dataset_path, n_nodes, write=False)
        
            singular_values_recv = []
            singular_vectors_recv = []
            P_i_t_recv = []
            for i in range(n_nodes):
                D_t, E_t, P_t = round_one(local_data[i], n_components)
                singular_values_recv.append(D_t)
                singular_vectors_recv.append(E_t)
                P_i_t_recv.append(P_t)
            # print('end round 1')
        
            P_i_hats_recv = []
            for i in range(n_nodes):
                P_hat, covMat_dist = round_two(singular_values_recv,
                                  singular_vectors_recv, P_i_t_recv[i], n_components)
                P_i_hats_recv.append(P_hat)
            # print('end round 2')
        
            projected_1 = np.concatenate(P_i_hats_recv, axis=0)
            
            # identity = np.array([[-1,0],[0,-1]])
            # identity = -1*np.identity(n_components)
            # projected_1 = np.matmul(projected_1,identity)
            
            dist_label_mat = np.concatenate(local_labels, axis=0)
            dist_label_mat = np.array(dist_label_mat)
        
            # plot(projected_1, dist_label_mat, 'distributed')
            # print('end test')
            X = data_mat
            X = X - np.mean(X, axis=0)
            X = np.array(X)
            # JD = TestEvaluationMetric2(projected_1, X, n_components)
            # print(JD)
            # plot(X, label_mat, 'dataset')    
            
            covMat = np.cov(np.transpose(X))
            ratio = TestEvaluationMetric3(X, projected_1, n_components)
            list_temp.append(ratio)
        ratio_dict[n_nodes] = list_temp
    
    ### central
    pca = PCA(n_components)
    projected_2 = pca.fit_transform(data_mat)
    # print(np.shape(projected_2))
    plot(projected_2, label_mat, 'central')
    
    # JD = TestEvaluationMetric2(projected_2, projected_1, n_components)
    # print(JD)
    
    
    X = data_mat
    X = X - np.mean(X, axis=0)
    X = np.array(X)
    JD = TestEvaluationMetric2(projected_1, X, n_components)
    # print(JD)
    plot(X, label_mat, 'dataset')    
    
    covMat = np.cov(np.transpose(X))
    TestEvaluationMetric3(covMat, covMat_dist, n_components)
    
    #################################################################
    """
    PLOTTING
    """
    for v in range(1, n_nodes_all):
        plt.plot(range(2, n_components_all), ratio_dict[v], label=v)
    plt.legend()
    plt.show()
    