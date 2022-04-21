import imp
import p2p_node
import numpy as np
from matplotlib import pyplot as plt


class PCANode(p2p_node.Node):
    def __init__(self, hostname, port, id, max_connections=-1, debug=False, t=None, data=None):
        super().__init__(hostname, port, id, max_connections, debug)

        self.n_components = t
        self.df = data.to_numpy()
        self.mean = None

        self.singular_values_recv = {}
        self.singular_vectors_recv = {}
        self.P_i_hats_recv = {}
        self.cov_mat = None

    
    def round_one(self):
        self.debug_print(f'Node.round1: ')
        self.mean = np.mean(self.df, axis=0)
        self.df -= self.mean
        self.U, self.S, V_T = np.linalg.svd(self.df, full_matrices=False)
        self.U, V_T = np.linalg.svd_flip(self.U, V_T)

        self.S = self.S[:self.n_components]
        self.V = np.transpose(V_T[:self.n_components])

        msg = {}
        #msg['singular_values'] = np.ndarray.tolist(self.S)
        msg['singular_vectors'] = np.ndarray.tolist(self.V)
        self.broadcast(msg)

    def round_two(self):
        self.g_cov_mat = np.zeros((self.n_components, self.n_components))
    
        for id, V_t in self.singular_vectors_recv.items():
            P_i_t = np.matmul(self.df, V_t) 
            cov_mat = np.matmul(np.transpose(P_i_t), P_i_t)
            self.g_cov_mat = np.add(self.g_cov_mat, cov_mat) 

        eigen_values, eigen_vectors = np.linalg.eig(self.g_cov_mat)
        eigen_stuff = list(zip(eigen_values, eigen_vectors))
        eigen_stuff = sorted(eigen_stuff, key=lambda x: x[0], reverse=True)
        sorted_eigen_values, sorted_eigen_vectors = zip(*eigen_stuff)
        sorted_eigen_vectors = np.array(sorted_eigen_vectors)
        sorted_eigen_vectors = sorted_eigen_vectors[:, :self.n_components]
        self.P_i_hat = np.matmul(P_i_t, sorted_eigen_vectors)
        P_i_hat = np.matmul(P_i_hat, np.transpose(sorted_eigen_vectors))



    def message_handler(self, sender_id, msg):
        super().message_handler(sender_id, msg)    

        if isinstance(msg, dict):
            for key, val in msg.items():
                if key == 'singular_values':
                    self.singular_values_recv[sender_id] = val #np.diag(val) #convert to mat when needed
                elif key == 'singular_vectors':
                    self.singular_vectors_recv[sender_id] = val
                elif key == 'P_i_hat':
                    self.P_i_hats_recv[sender_id] = val
                else:
                    self.debug_print(f'Node.message_handler: Unknown key {key}')

        # TODO: add more robust comparisions? 
        if len(self.singular_vectors_recv) >= len(self.connections): #got V from all nodes
            self.round_two()
        
        if len(self.P_i_hats_recv) >= len(self.connections): 
            self.P_hat = np.concatenate(self.P_i_hats_recv.values(), axis=0)

    