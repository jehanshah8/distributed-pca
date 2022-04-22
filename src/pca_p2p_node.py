import imp
import p2p_node
import numpy as np
from matplotlib import pyplot as plt


class PCANode(p2p_node.Node):
    def __init__(self, hostname, port, id, max_connections=-1, debug=False, t=None, data=None):
        super().__init__(hostname, port, id, max_connections, debug)

        self.n_components = t
        self.local_data = data.to_numpy()
        self.mean = None

        self.singular_values_recv = {}
        self.singular_vectors_recv = {}
        self.projected_data_recv = {}
        self.cov_mat = None

    def round_one(self):
        self.debug_print(f'Node.round1: ')
        self.mean = np.mean(self.local_data, axis=0)
        self.local_data -= self.mean

        U, D, E_T = np.linalg.svd(self.local_data, full_matrices=True)

        D_t = np.zeros((np.shape(self.local_data)[0], self.n_components))
        for i in range(self.n_components):
            D_t[i][i] = D[i]

        E_T_t = E_T[:self.n_components]
        self.local_data = np.matmul(U, D_t)  # D_i_t
        self.local_data = np.matmul(self.local_data, E_T_t)

        msg = {}
        msg['singular_values'] = np.ndarray.tolist(D[:self.n_components])
        msg['singular_vectors'] = np.ndarray.tolist(np.transpose(E_T_t))
        self.broadcast(msg)

    def round_two(self):
        g_cov_mat = np.zeros((np.shape(self.local_data)[1], np.shape(self.local_data)[1]))

        # for data from each node
        for id in self.connections.keys():

            D_t = np.zeros((np.shape(self.local_data)[0], self.n_components))
            
            for j in range(self.n_components):
                D_t[j][j] = self.singular_values_recv[id][j]
            
            E_t = self.singular_vectors_recv[id]
        
            cov_mat = np.matmul(E_t, np.transpose(D_t))
            cov_mat = np.matmul(cov_mat, D_t)
            cov_mat = np.matmul(cov_mat, np.transpose(E_t))

            g_cov_mat = np.add(g_cov_mat, cov_mat)

        eigen_values, eigen_vectors = np.linalg.eig(g_cov_mat)
        eigen_stuff = list(zip(eigen_values, eigen_vectors))
        eigen_stuff = sorted(eigen_stuff, key=lambda x: x[0], reverse=True)
        sorted_eigen_values, sorted_eigen_vectors = zip(*eigen_stuff)
        sorted_eigen_vectors = np.array(sorted_eigen_vectors)
        sorted_eigen_vectors = sorted_eigen_vectors[:, :self.n_components]

        self.local_data = np.matmul(self.local_data, sorted_eigen_vectors)
        #TODO: remove below if you only want t columns 
        self.local_data = np.matmul(self.local_data, np.transpose(sorted_eigen_vectors)) 

        msg = {}
        msg['local_projected_data'] = np.ndarray.tolist(self.local_data)
        self.broadcast(msg)

    def message_handler(self, sender_id, msg):
        super().message_handler(sender_id, msg)

        if isinstance(msg, dict):
            for key, val in msg.items():
                if key == 'singular_values':
                    # np.diag(val) #convert to mat when needed
                    self.singular_values_recv[sender_id] = val
                elif key == 'singular_vectors':
                    self.singular_vectors_recv[sender_id] = val
                elif key == 'local_projected_data':
                    self.projected_data_recv[sender_id] = val
                else:
                    self.debug_print(
                        f'Node.message_handler: Unknown key {key}')

        # TODO: add more robust comparisions?
        if len(self.singular_vectors_recv) >= len(self.connections):  # got V from all nodes
            self.round_two()

        if len(self.projected_data_recv) >= len(self.connections):
            approx_global_data = np.concatenate(self.projected_data_recv.values(), axis=0)

            if self.id == '0':
                #write to file

                # run metrics
                pass
                