import socket
import src.pca_coordinator as coordinator
import src.pca_node as node


class DistributedPCA():
    def __init__(self, n_nodes, n_mal_nodes=0, mal_type=0, coordinator='naive'):
        self._hostname = socket.gethostbyname(socket.gethostname())
        self._port = 5050

        self.n_nodes = n_nodes
        self.n_mal_nodes = n_mal_nodes
        self.nodes = []
        if coordinator == 'naive':
            self.coordinator = coordinator.NaivePCACoordinator(n_nodes=n_nodes)

        self.coordinator.bind((self._hostname, self._port))
        self.coordinator.start()

        for i in range(n_nodes - n_mal_nodes):
            self.nodes.append(node.PCANode(id=i))
            self.nodes[i].connect(self._hostname, self._port)

        if mal_type == 1:
            for i in range(n_mal_nodes):
                self.nodes.append(node.MalPCANode(id=i))
                self.nodes[i].connect(self._hostname, self._port)

    def fit_transform(self, dataset_path, n_components):
        coordinator.run(dataset_path, n_components)
        for v in self.nodes:
            v.run()
