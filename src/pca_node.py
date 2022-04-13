import sys
import socket
import pandas as pd

import src.client as client
import src.utils as utils


class PCANode(client.Client):
    def __init__(self, header=128, format='utf-8', disconnect_msg='Disconnecting'):
        super().__init__(header, format, disconnect_msg)

    def run(self):
        round_one_info = self.request("round_one")
        data_mat, label_mat = utils.load_dataset(
            utils.get_local_dataset_path(dataset_path))


class MalPCANode1(client.Client):
    pass


if __name__ == '__main__':
    pass
