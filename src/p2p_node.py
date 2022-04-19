import threading
import socket
import time
import random
import hashlib

class Connection(threading.Thread):
    """This class is used to represent a connection for the Node class"""

    def __init__(self, main_node, sock, sock_id, hostname, port, encoding='utf-8'):
        super(Connection, self).__init__()
        self.main_node = main_node
        self.hostname = hostname
        self.port = port
        self.sock = sock
        self.sock_id = str(sock_id)

        # End of transmission character for the network streaming messages.
        self.EOT_CHAR = 0x04.to_bytes(1, 'big')

        # Byte encoding format
        self.encoding = encoding

        self.received_messages = []

        self.sock.settimeout(10.0)

        self.terminate_flag = threading.Event()

        self.main_node.debug_print(
            f'Connection.init: Started with client {self.sock_id} | {self.hostname} : {self.port}')

    def send(self, msg):
        try:
            self.sock.sendall(msg.encode(self.encoding_type) + self.EOT_CHAR)

        except Exception as e:  # When sending is corrupted, close the connection
            self.main_node.debug_print(
                'Connection.send: Error sending data to node: ' + str(e))
            self.stop()  # Stopping node due to failure

        # TODO: encrypted communication? signed messages and authentication?

    def stop(self):
        self.terminate_flag.set()
        self.main_node.debug_print(
            f'Connection.stop: Terminate flag = {self.terminate_flag.is_set()}')

    def run(self):
        # Listens to messages and stores them in a buffer queue
        while not self.terminate_flag.is_set():
            chunk = b''
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                self.main_node.debug_print('Connection.receive: Timeout')
            except Exception as e:
                self.terminate_flag.set()  # Exception occurred terminating the connection
                self.main_node.debug_print(
                    'Connection.receive: Unexpected error')
                self.main_node.debug_print(e)

            if chunk != b'':
                buffer += chunk
                eot_pos = buffer.find(self.EOT_CHAR)

                while eot_pos > 0:
                    self.received_messages.append(
                        buffer[:eot_pos].decode(self.encoding))
                    buffer = buffer[eot_pos + 1:]

                    # self.main_node.message_count_recv += 1 # decide how I want main node to fetch messages from queue
                    eot_pos = buffer.find(self.EOT_CHAR)

            time.sleep(0.01)

        # IDEA: Invoke (event) a method in main_node so the user is able to send a bye message to the node before it is closed?
        self.sock.settimeout(None)
        self.sock.close()
        # Fixed issue #19: Send to main_node when a node is disconnected. We do not know whether it is inbounc or outbound.
        self.main_node.node_disconnected(self)
        self.main_node.debug_print('Connection.run: Stopped')


class Node(threading.Thread):
    """
        This class represents a node in a p2p network.
        It can hanle requests from multiple nodes it is connected to
        It can broadcast messages to all nodes it is connected to 
    """

    def __init__(self, hostname, port, id, max_connections=0, debug=False):
        super(Node, self).__init__()

        self.hostname = hostname
        self.port = port

        if id == None:
            self.id = self.generate_id()

        else:
            self.id = str(id)  # Make sure the ID is a string!

        # List of the Connection objects that connect TO this node (connection->this)
        self.inbound_connections = []
        # List of the Connections objects this node connects to (this->connection)
        self.outbound_connections = []
        # in general, items in inboud_connections = items in outbound_connections

        # A list of nodes that should be reconnected to whenever the connection was lost
        self.reconnect_to_nodes = []

        self.terminate_flag = threading.Event()

        # Message counters to make sure everyone is able to track the total messages
        self.message_count_send = 0
        self.message_count_recv = 0
        self.message_count_rerr = 0

        # Connection limit of inbound nodes (nodes that connect to us)
        self.max_connections = max_connections

        # Debugging on or off!
        self.debug = debug

        # Start the TCP/IP server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.start_server()

    def generate_id(self):
        """Generates a unique ID for each node."""
        id = hashlib.sha512()
        t = self.host + str(self.port) + str(random.randint(1, 99999999))
        id.update(t.encode('ascii'))
        return id.hexdigest()

    def debug_print(self, message):
        """When the debug flag is set to True, all debug messages are printed in the console."""
        if self.debug:
            print(f'[{self.id}] DEBUG {message}')

    def start_server(self):
        """Starts the server to handle inbound connections """
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.hostname, self.port))
        self.sock.settimeout(10.0)
        self.sock.listen(1)
        print(f'[{self.id}] Initialized node on {self.hostname} : {self.port}')

    def send(self, conn, msg):
        self.message_count_send = self.message_count_send + 1
        if conn in self.inbound_nodes or conn in self.outbound_nodes:
            conn.send(msg)
        else:
            self.debug_print(
                'Node.send: Could not send the data, node is not found!')


    def broadcast(self, data, exclude=[]):
        self.message_count_send = self.message_count_send + 1
        for n in self.nodes_inbound:
            if n in exclude:
                self.debug_print("Node send_to_nodes: Excluding node in sending the message")
            else:
                self.send_to_node(n, data)

        for n in self.nodes_outbound:
            if n in exclude:
                self.debug_print("Node send_to_nodes: Excluding node in sending the message")
            else:
                self.send_to_node(n, data)

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
