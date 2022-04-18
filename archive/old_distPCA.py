from operator import matmul
import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip


class LocalPCA: 
    def __init__(self, n_components=None): 
        self.pca = PCA(n_components)

    def fit(self, X, Y=None):
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, Vt = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        components_ = Vt
        
class GlobalPCA: 
    def __init__(self, n_components=None): 
        self.n_components = n_components
    
    def fit(self, X, y=None):     
        pass

    def fit_transform(self, nodes):
        assert nodes.len() > 1

        local_cov_mats = []
        for v in nodes: 
            local_cov_mats.append(matmul(v.S, v.Vt))

        global_cov_mat = np.vstack(local_cov_mats)

        # Algorithm 1 lines 11 - 12
        U, S, Vt = np.linalg.svd(global_cov_mat, full_matrices=True) #TODO: full mat?
        U = U[:, : self.n_components_]

        # Algorithm 1 lines 11 - 12
        for v in nodes: 
            local_cov_mats.append(matmul(v.S, v.Vt))
        # P_new = X * V = U * S * Vt * V = U * S
        U *= S[: self.n_components_]

        return U


