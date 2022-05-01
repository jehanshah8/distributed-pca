from sklearn.metrics import explained_variance_score
import p2p_node
import numpy as np
import time
import socket
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

class PCANode(p2p_node.Node):
    def __init__(self, hostname, port, id, max_connections=-1, debug=False, t=None, data=None):
        super().__init__(hostname, port, id, max_connections, debug)

        self.n_components = t
        self.local_data = data
        self.mean = None

        self.round_one_complete = False
        self.E_t = None
        
        self.singular_values_recv = {}
        self.singular_vectors_recv = {}
        self.data_lengths_recv = {}
        
        self.reduce_dim = False
        self.round_two_started = False
        self.round_two_complete = False
        
        self.g_cov_mat = None
        self.total_variance = None
        self.projected_data_recv = {}

        
        self.projected_global_data = None
        self.pca_complete = False

    def do_PCA(self, data, t, reduce_dim=False):
        self.debug_print(f'Node.do_PCA: started')
        self.n_components = t
        self.local_data = data
        self.reduce_dim = reduce_dim
        self.do_round_one()
        #self.debug_print(f'Node.do_PCA: finished')

    def reset(self):
        self.round_one_complete = False

        self.singular_values_recv = {}
        self.singular_vectors_recv = {}
        self.data_lengths_recv = {}
        
        self.reduce_dim = False
        self.round_two_started = False
        self.round_two_complete = False
        
        self.g_cov_mat = None
        self.total_variance = None
        self.projected_data_recv = {}

        self.projected_global_data = None
        self.pca_complete = False

    #def get_transformed_data(self):
    #    #self.debug_print(f'Node.get_transformed_data: started')
    #    if self.pca_complete:
    #        #self.debug_print(f'Node.get_transformed_data: finished')
    #        return self.projected_global_data
    #    else: 
    #        return None
#
    #def get_cov_mat(self):
    #    #self.debug_print(f'Node.get_cov_mat: started')
    #    if self.round_two_complete:
    #        #self.debug_print(f'Node.get_transformed_data: finished')
    #        return self.g_cov_mat
    #    else: 
    #        return None

    def do_round_one(self):
        self.debug_print(f'Node.do_round_one: started')
        mean = np.mean(self.local_data, axis=0)
        self.local_data -= mean

        U, D, E_T = np.linalg.svd(self.local_data, full_matrices=True)

        D_t = np.zeros((np.shape(self.local_data)[0], self.n_components))
        for i in range(self.n_components):
            D_t[i][i] = D[i]

        E_t_T = E_T[:self.n_components]
        self.local_data = np.matmul(U, D_t)  # D_i_t
        self.local_data = np.matmul(self.local_data, E_t_T)

        self.E_t = np.transpose(E_t_T)

        msg = {}
        msg['singular_values'] = np.ndarray.tolist(D[:self.n_components])
        msg['singular_vectors'] = np.ndarray.tolist(self.E_t)
        msg['data_length'] = np.shape(self.local_data)[0]
        
        self.round_one_complete = True
        self.debug_print(f'Node.do_round_one: finished')
        
        self.broadcast(msg)
        

    def do_round_two(self):
        #self.round_two_started = True
        self.debug_print(f'Node.do_round_two: started')
        #self.debug_print(np.shape(self.local_data))
        g_cov_mat = np.zeros(
            (np.shape(self.local_data)[1], np.shape(self.local_data)[1]))

        # for data from each node
        for id in self.connections.keys():

            D_t = np.zeros((np.shape(self.local_data)[0], self.n_components))

            for j in range(self.n_components):
                D_t[j][j] = self.singular_values_recv[id][j]

            E_t = self.singular_vectors_recv[id]

            cov_mat = np.matmul(E_t, np.transpose(D_t))
            cov_mat = np.matmul(cov_mat, D_t)
            cov_mat = np.matmul(cov_mat, np.transpose(E_t))
            cov_mat = cov_mat / (self.data_lengths_recv[id] - 1)
            g_cov_mat = np.add(g_cov_mat, cov_mat)

        self.g_cov_mat = g_cov_mat
        U, D, E_T = np.linalg.svd(g_cov_mat, full_matrices=True)

        # projecting data onto the reduced (n_components) dimensional space
        self.local_data = np.matmul(
            self.local_data, np.transpose(E_T[:self.n_components]))

        # adding below means projecting data onto original dimensional space
        if not self.reduce_dim:
            self.local_data = np.matmul(self.local_data, E_T[:self.n_components])

        msg = {}
        msg['local_projected_data'] = np.ndarray.tolist(self.local_data)
        
        self.debug_print(f'Node.do_round_two: finished')
        self.round_two_complete = True

        self.broadcast(msg)
        
    def calc_total_variance(self):
        U, D, E_T = np.linalg.svd(self.projected_global_data, full_matrices=True)
        explained_variance = (D**2) / (np.shape(self.projected_global_data)[0] - 1)
        self.total_variance = explained_variance.sum()

    def message_handler(self, sender_id, msg):
        super().message_handler(sender_id, msg)

        if isinstance(msg, dict):
            for key, val in msg.items():
                if key == 'singular_values':
                    # np.diag(val) #convert to mat when needed
                    self.singular_values_recv[sender_id] = val
                elif key == 'singular_vectors':
                    self.singular_vectors_recv[sender_id] = val
                elif key == 'data_length':
                    self.data_lengths_recv[sender_id] = val
                elif key == 'local_projected_data':
                    self.projected_data_recv[sender_id] = val
                else:
                    self.debug_print(
                        f'Node.message_handler: Unknown key {key}')

        # TODO: add more robust comparisions?
        if len(self.singular_vectors_recv) >= len(self.connections) and not self.round_two_started:  # got V from all nodes
            self.round_two_started = True
            while not self.round_one_complete:
                time.sleep(5)
            
            self.do_round_two()

        if len(self.projected_data_recv) >= len(self.connections) and self.round_two_complete:
            self.debug_print('got P_hat from all nodes')
            projected_data_recv_list = []
            projected_data_recv_list.append(self.local_data)
            [projected_data_recv_list.append(p) for  p in self.projected_data_recv.values()]

            self.projected_global_data = np.concatenate(projected_data_recv_list, axis=0)

            self.calc_total_variance()

            #self.debug_print(np.shape(self.projected_global_data))
            self.debug_print('PCA over')
            self.pca_complete = True


class SecurePCA1(PCANode):
    def __init__(self, hostname, port, id, max_connections=-1, debug=False, t=None, data=None):
        super().__init__(hostname, port, id, max_connections, debug, t, data)

    def do_round_two(self):
        #self.round_two_started = True
        self.debug_print(f'Node.do_round_two: started')
        #self.debug_print(np.shape(self.local_data))
        g_cov_mat = np.zeros(
            (np.shape(self.local_data)[1], np.shape(self.local_data)[1]))

        similarities = {}
        # for data from each node
        for id in self.connections.keys():

            D_t = np.zeros((np.shape(self.local_data)[0], self.n_components))

            for j in range(self.n_components):
                D_t[j][j] = self.singular_values_recv[id][j]

            E_t = self.singular_vectors_recv[id]

            #security enhancement
            #print(type(self.E_t), np.shape(self.E_t))
            #print(type(E_t), np.shape(E_t))
            #similarities[id] = cosine_similarity(np.transpose(self.E_t), np.transpose(E_t)) 
            
            
            similarities[id] = []
            A = np.transpose(self.E_t)
            B = np.transpose(E_t)
            for i in range(self.n_components):
                similarities[id].append(np.dot(A[i],B[i]) / (norm(A[i])*norm(B[i])))
            #self.debug_print(f'Cosine similarity with {id}:, {similarities[id]}')
            #


            cov_mat = np.matmul(E_t, np.transpose(D_t))
            cov_mat = np.matmul(cov_mat, D_t)
            cov_mat = np.matmul(cov_mat, np.transpose(E_t))
            cov_mat = cov_mat / (self.data_lengths_recv[id] - 1)
            g_cov_mat = np.add(g_cov_mat, cov_mat)

        self.g_cov_mat = g_cov_mat
        U, D, E_T = np.linalg.svd(g_cov_mat, full_matrices=True)

        # projecting data onto the reduced (n_components) dimensional space
        self.local_data = np.matmul(
            self.local_data, np.transpose(E_T[:self.n_components]))

        # adding below means projecting data onto original dimensional space
        if not self.reduce_dim:
            self.local_data = np.matmul(self.local_data, E_T[:self.n_components])

        msg = {}
        msg['local_projected_data'] = np.ndarray.tolist(self.local_data)
        
        self.debug_print(f'Node.do_round_two: finished')
        self.round_two_complete = True

        self.broadcast(msg)



class MalPCANode(PCANode):
    def __init__(self, hostname, port, id, max_connections=-1, debug=False, t=None, data=None):
        super().__init__(hostname, port, id, max_connections, debug, t, data)
        self.attack_strategy = 0

    def set_attack_strategy(self, a): 
        self.attack_strategy = a

    def do_round_one(self):
        self.debug_print(f'MalCPANode.do_round_one: started')
        
        ## this is the logic
        mean = np.mean(self.local_data, axis=0)
        self.local_data -= mean

        U, D, E_T = np.linalg.svd(self.local_data, full_matrices=True)

        D_t = np.zeros((np.shape(self.local_data)[0], self.n_components))
        for i in range(self.n_components):
            D_t[i][i] = D[i]

        E_t_T = E_T[:self.n_components]

         ## DEFINE ATTACKS HERE:
        if self.attack_strategy == 1:
            # randomize the values in the singular vectors 
            shape = np.shape(E_t_T)
            E_t_T = np.random.rand(shape[0], shape[1])
            
            #bounds = []
            #for v in E_t_T:
            #    min = np.min(v)
            #    max = np.max(v)
            #    v = np.random(size=v.size())

        elif self.attack_strategy == 2:
            # pick the least significant singular vectors first
            E_t_T = E_T
            E_t_T = np.flip(E_t_T, axis=0)
            E_t_T = E_t_T[:self.n_components]
        
        elif self.attack_strategy == 3:
            # make singular vectors perpendicular
            for v in E_t_T:
                x = -v[:-1].sum() / v[-1] 
                u = np.ones_like(v)
                #u = u.astype(np.float32)
                u[-1] = x
                v = u

        self.local_data = np.matmul(U, D_t)  # D_i_t
        self.local_data = np.matmul(self.local_data, E_t_T)
        ## end logic


        # leave below untouched. Just sending stuff
        msg = {}
        msg['singular_values'] = np.ndarray.tolist(D[:self.n_components])
        msg['singular_vectors'] = np.ndarray.tolist(np.transpose(E_t_T))
        msg['data_length'] = np.shape(self.local_data)[0]
        
        self.round_one_complete = True
        self.debug_print(f'MalCPANode.do_round_one: finished')
        
        self.broadcast(msg)
