import sys
import src.utils as utils

from operator import matmul
import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip


class LocalPCA: 
    def __init__(self, n_components=None, id=-1): 
        self.pca = PCA(n_components)
        self.id = id
        self.df = None

    def fit(self, X, Y=None):
        self.df = X

        self.mean = np.mean(X, axis=0)
        X -= self.mean

        U, S, Vt = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        return U, S, Vt


if __name__ == "__main__":
    if len(sys.argv) == 4:
        n_nodes = int(sys.argv[1])
        dataset_path = sys.argv[2]
        n_components = sys.argv[3]

    else:
        n_nodes = 3
        dataset_path = 'iris_with_cluster.csv'
        n_components = 4

    localPCAs = []
    for i in range(n_nodes):
        data_mat, label_mat = utils.load_dataset(utils.get_local_dataset_path(dataset_path, i))
        localPCAs.append(LocalPCA(n_components=n_components, id=i))
        localPCAs[i].fit(data_mat)