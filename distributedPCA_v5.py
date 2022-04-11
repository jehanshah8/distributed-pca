# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:30:27 2022

@author: Sai Mudumba
"""
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


def DistributedPCA(local_datasets, t, sample_size, samples_per_node):
    
    D1t, E1t, P1t = localPCA(local_datasets[0], t)
    D2t, E2t, P2t = localPCA(local_datasets[1], t)
    D3t, E3t, P3t = localPCA(local_datasets[2], t)
    D4t, E4t, P4t = localPCA(local_datasets[3], t)
    D5t, E5t, P5t = localPCA(local_datasets[4], t)
    D6t, E6t, P6t = localPCA(local_datasets[5], t)
    D7t, E7t, P7t = localPCA(local_datasets[6], t)
    D8t, E8t, P8t = localPCA(local_datasets[7], t)
    D9t, E9t, P9t = localPCA(local_datasets[8], t)
    D10t, E10t, P10t = localPCA(local_datasets[9], t)
    
    S1t = Covariance(D1t, E1t, P1t)
    S2t = Covariance(D2t, E2t, P2t)
    S3t = Covariance(D3t, E3t, P3t)
    S4t = Covariance(D4t, E4t, P4t)
    S5t = Covariance(D5t, E5t, P5t)
    S6t = Covariance(D6t, E6t, P6t)
    S7t = Covariance(D7t, E7t, P7t)
    S8t = Covariance(D8t, E8t, P8t)
    S9t = Covariance(D9t, E9t, P9t)
    S10t = Covariance(D10t, E10t, P10t)
    
    
    St = S1t+S2t+S3t+S4t+S5t+S6t+S7t+S8t+S9t+S10t
    print(St)
    v,w=eig(St)
    # print(v,w)
    Ui, di, EiT = np.linalg.svd(St, full_matrices=True)
    print(Ui, di, EiT)
    
    P1_hat = GlobalPCA(P1t, EiT)
    P2_hat = GlobalPCA(P2t, EiT)
    P3_hat = GlobalPCA(P3t, EiT)
    P4_hat = GlobalPCA(P4t, EiT)
    P5_hat = GlobalPCA(P5t, EiT)
    P6_hat = GlobalPCA(P6t, EiT)
    P7_hat = GlobalPCA(P7t, EiT)
    P8_hat = GlobalPCA(P8t, EiT)
    P9_hat = GlobalPCA(P9t, EiT)
    P10_hat = GlobalPCA(P10t, EiT)
    
    P_hat = [P1_hat, P2_hat, P3_hat, P4_hat, P5_hat, P6_hat, P7_hat, P8_hat, P9_hat, P10_hat]
    
    P_hat_T = np.zeros((t,sample_size))
    counter = 0
    for i in range(0, int(sample_size), int(samples_per_node)):
        P_hat_T[:,i:i+int(samples_per_node)] = np.transpose(P_hat[counter])
        # print(P_hat_T)
        counter+=1
    
    P_hat = np.transpose(P_hat_T)
    # print(P_hat.shape)
    # print(P_hat)

    plt.scatter(-P_hat[:, 1], P_hat[:, 0],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar();
    plt.show()
    
    return 

def localPCA(Pi, t):
    Pi = Pi - np.mean(Pi, axis=0)
    Ui, di, EiT = np.linalg.svd(Pi, full_matrices=True)
    Di = np.zeros((Pi.shape[0], Pi.shape[1]))
    Di[:Pi.shape[1], :Pi.shape[1]] = np.diag(di)
    Pi = Ui.dot(Di.dot(EiT))
    Ei = np.transpose(EiT)
    Di_t= np.zeros((Pi.shape[0], Pi.shape[1]))
    Di_t[:2,:2] = np.diag(di[0:2])
    
    Ei_t = np.zeros((Pi.shape[1],2))
    Ei_t[:,0] = Ei[0]
    Ei_t[:,1] = Ei[1]
    Pi_t = Ui.dot(Di_t.dot(Ei_t))
    print(Pi_t.shape)
    return Di_t, Ei_t, Pi_t

def Covariance(Di_t, Ei_t, Pi_t):
    Si_t = np.matmul(np.transpose(Pi_t),Pi_t)
    return Si_t

def GlobalPCA(Pi_t, EiT):
    Pi_hat = Pi_t.dot(EiT.dot(np.transpose(EiT)))
    return Pi_hat
    
    
digits = load_digits()
X = digits.data
Y = digits.target

nodes = 10
sample_size = X.shape[0]

samples_per_node = np.ceil(sample_size/nodes)
counter = 0

X_dist = []#np.zeros((nodes,1))
Y_dist = []#np.zeros((nodes,1))
for i in range(0, int(sample_size), int(samples_per_node)):
    counter += 1
    # Without replacement
    X_new = X[i:i+int(samples_per_node)]
    Y_new = Y[i:i+int(samples_per_node)]

    print(X_new.shape)
 
    X_dist.append(X_new)
    Y_dist.append(Y_new)

X_dist = np.array(X_dist, dtype=object)
Y_dist = np.array(Y_dist, dtype=object)

DistributedPCA(X_dist, 2, sample_size, samples_per_node)
