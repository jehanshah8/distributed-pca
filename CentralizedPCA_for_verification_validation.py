# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 22:13:41 2022

@author: Sai Mudumba
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
PCA for visualization
"""
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
X = X - np.mean(X, axis=0)
Y = digits.target

### METHOD 1: Use SVD to Find Projected Vector Onto 2-dimensions
U, s, Vt = np.linalg.svd(X, full_matrices=True)
pc = np.matmul(Vt[0:2],np.transpose(X))
# pc = np.matmul(X,Vt[:,0:2]) # this expression does not get the right answer
pc = np.transpose(pc)
pc1 = pc[:,0]
pc2 = pc[:,1]
plt.scatter(-pc1, -pc2,
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('Method 1: Using SVD and Getting First 2 Columns of V')
plt.colorbar();
plt.show()

# cumulative = [sum(s[0:i+1])/sum(s) for i in range(64)]
# plt.plot(cumulative)
# plt.show()

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
# pca = PCA().fit(digits.data)
# plt.plot(np.arange(64)+1, np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');