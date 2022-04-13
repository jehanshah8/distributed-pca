import sys
import socket
import time

import src.server as server


class NaivePCACoordinator(server.Server):
    def __init__(self, header=128, format='utf-8', disconnect_msg='Disconnecting', n_nodes=1):
        super().__init__(header, format, disconnect_msg, n_nodes)
        self.dataset_path = None
        self.n_components = None

    def handle_client(self, conn, addr):
        print(f"[{addr}] New connection")

        connected = True
        while connected:
            msg = self.receieve()

            print(f"[{addr}] {msg}")

            if msg == self.disconnect_msg:
                connected = False
            elif msg == "dataset":
                while self.dataset_path == None:
                    time.sleep(0.1)
                else:
                    self.send(conn, self.dataset_path)
            elif msg == "n_components":
                while self.n_components == None:
                    time.sleep(0.1)
                else:
                    self.send(conn, str(self.n_components))
            elif msg == "round_one":
                while self.round_one == False:
                    time.sleep(0.1)
                else:
                    self.send(conn, " ")  # TODO
            elif msg == "round_two":
                while self.round_two == False:
                    time.sleep(0.1)
                else:
                    self.send(conn, " ")  # TODO

        conn.close()
        print(f"[{addr}] Disconnected")

    def run(self, dataset_path, n_components):
        self.dataset_path = dataset_path
        self.n_components = n_components


if __name__ == '__main__':
    pass
