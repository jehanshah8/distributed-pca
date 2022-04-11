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
    S1t = Covariance(D1t, E1t, P1t)
    St = S1t
    Ui, di, EiT = np.linalg.svd(St, full_matrices=True)    
    P1_hat = GlobalPCA(P1t, EiT)
    P_hat = [P1_hat]
    P_hat_T = np.zeros((t,sample_size))
    counter = 0
    for i in range(0, int(sample_size), int(samples_per_node)):
        P_hat_T[:,i:i+int(samples_per_node)] = np.transpose(P_hat[counter])
        counter+=1
    P_hat = np.transpose(P_hat_T)

    plt.scatter(-P_hat[:, 1], P_hat[:, 0],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title('Method: Distributed PCA with nodes = 1')
    plt.colorbar();
    plt.show()
    
    return 

def localPCA(Pi, t):
    Pi = Pi - np.mean(Pi, axis=0)
    Pi = np.array(Pi,dtype=np.float)
    Ui, di, EiT = np.linalg.svd(Pi, full_matrices=True)
    Di = np.zeros((Pi.shape[0], Pi.shape[1]))
    Di[:Pi.shape[1], :Pi.shape[1]] = np.diag(di)
    Pi = Ui.dot(Di.dot(EiT))
    Ei = np.transpose(EiT)
    Di_t= np.zeros((Pi.shape[0], Pi.shape[1]))
    Di_t[:t,:t] = np.diag(di[0:t])
    
    Ei_t = np.zeros((Pi.shape[1],t))
    for j in range(0,t):
        Ei_t[:,j] = Ei[j]
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

nodes = 1
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

## METHOD 2: USe PCA function from sklearn library to find Projected Vector onto 2-dimensions
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('Method 2: Using PCA Tool from Sklearn')
plt.colorbar();
plt.show()