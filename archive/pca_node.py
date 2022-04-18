import sys
import socket
import pandas as pd

import src.client as client
import src.utils as utils


class PCANode():
    """
        Each node has a list of nodes that it is connected to. 
        For each, it has a corresponding serever AND client
    """

    def __init__(self):
        self.peers = []
        

    def addConnection(self, peer):
        pass


    def run(self):
        round_one_info = self.request("round_one")
        data_mat, label_mat = utils.load_dataset(
            utils.get_local_dataset_path(dataset_path))


class MalPCANode1():
    pass


if __name__ == '__main__':
    pass
