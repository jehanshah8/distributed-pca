import imp
import threading
import socket
import time


class Connection(threading.Thread):
    """This class is used to represent a connection for the Node class"""

    def __init__(self, main_node, sock, sock_id, hostname, port, encoding="utf-8"):
        super(Connection, self).__init__()
        self.main_node = main_node
        self.hostname = hostname
        self.port = port
        self.sock = sock
        self.sock_id = str(sock_id)
        self.encoding = encoding
        self.received_messages = []

        # End of transmission character for the network streaming messages.
        self.EOT_CHAR = 0x04.to_bytes(1, 'big')

        self.sock.settimeout(10.0)

        self.main_node.debug_print(
            "Connection: Started with client (" + self.id + ") '" + self.host + ":" + str(self.port) + "'")

        self.terminate_flag = threading.Event()

    def send(self, msg):
        try:
            self.sock.sendall(msg.encode(self.encoding_type) + self.EOT_CHAR)

        except Exception as e:  # When sending is corrupted, close the connection
            self.main_node.debug_print(
                "Connection.send: Error sending data to node: " + str(e))
            self.stop()  # Stopping node due to failure

        # TODO: encrypted communication? signed messages and authentication?

    def stop(self):
        self.terminate_flag.set()

    def run(self):
        # Listens to messages and stores them in a buffer queue
        while not self.terminate_flag.is_set():
            chunk = b''
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                self.main_node.debug_print("Connection.receive: Timeout")
            except Exception as e:
                self.terminate_flag.set()  # Exception occurred terminating the connection
                self.main_node.debug_print(
                    'Connection.receive: Unexpected error')
                self.main_node.debug_print(e)

            if chunk != b'':
                buffer += chunk
                eot_pos = buffer.find(self.EOT_CHAR)

                while eot_pos > 0:
                    self.received_messages.append(buffer[:eot_pos].decode(self.encoding))
                    buffer = buffer[eot_pos + 1:]

                    # self.main_node.message_count_recv += 1 # decide how I want main node to fetch messages from queue
                    eot_pos = buffer.find(self.EOT_CHAR)

            time.sleep(0.01)

        # IDEA: Invoke (event) a method in main_node so the user is able to send a bye message to the node before it is closed?
        self.sock.settimeout(None)
        self.sock.close()
        # Fixed issue #19: Send to main_node when a node is disconnected. We do not know whether it is inbounc or outbound.
        self.main_node.node_disconnected(self)
        self.main_node.debug_print("Connection.run: Stopped")


class Node(threading.Thread):
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
