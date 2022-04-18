from threading import Thread
import src.client as cl
import src.server as srv


class Connection(Thread):
    """This class is used to represent a connection for the Node class"""

    def __init__(self, hostname, port, uid):
        self.hostname = hostname
        self.port = port
        self.uid = uid
        #self.connections = []


class Node(Thread):
    """
        This class represents a node in a p2p network.
        It can hanle requests from multiple nodes it is connected to
        It can broadcast messages to all nodes it is connected to 
    """

    def __init__(self, hostname, port, id=None):
        # List of the Connections??? of nodes that this node serves
        self.inbound_connection = []

        # List of the Connections??? of the nodes that this node connects TO
        self.outbound_connections = []

    def start():
        """Starts the server to handle inbound connections """
        pass

    def handle_inbound_connection(self):
        pass

    def add_outbound_connection(hostname, port):
        """Connects to the node at given hostname and port"""
        pass

    def remove_outbound_connection(self, peer):
        pass

    def disconnect_all(self):
        pass

    def request_all(self, msg):
        pass

    def stop():
        """This removes all outbound connections, inbound connections and gracefully exits"""
        pass

    def run():
        pass
