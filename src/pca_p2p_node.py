import p2p_node

class PCANode(p2p_node.Node):
    def __init__(self, hostname, port, id, max_connections=-1, debug=False):
        super().__init__(hostname, port, id, max_connections, debug)

