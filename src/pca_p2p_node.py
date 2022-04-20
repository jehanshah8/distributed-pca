import imp
import p2p_node
import numpy as np


class PCANode(p2p_node.Node):
    def __init__(self, hostname, port, id, max_connections=-1, debug=False, t=None, data=None):
        super().__init__(hostname, port, id, max_connections, debug)

        self.n_components = t
        self.df = data
        self.mean = None
        


        self.singular_values_recv = {}
        self.singular_vectors_recv = {}
        self.P_i_hats_recv = {}
        self.cov_mat = None

    
    def round_one(self):
        self.mean = np.mean(self.df, axis=0)
        self.df -= self.mean
        self.left_singular_vectors, self.singular_values, self.right_singular_vectors = np.linalg.svd(self.df, full_matrices=False)
        self.left_singular_vectors, self.right_singular_vectors = np.linalg.svd_flip(self.left_singular_vectors, self.right_singular_vectors)

        self.singular_values = self.singular_values[:self.n_components]
        self.right_singular_vectors = np.transpose(self.right_singular_vectors[:self.n_components])
        self.debug_print(f'Node.round1: ')


        msg = {}
        msg['singular_values'] = np.ndarray.tolist(self.singular_values)
        msg['singular_vectors'] = np.ndarray.tolist(self.right_singular_vectors)
        self.broadcast(msg)

    def round_two(self):
        pass

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

        # TODO: add more robust compare? 
        if len(self.P_i_hats_recv) >= len(self.connections):
            #self.P_hat = 
            pass