import threading
import socket
import time
import random
import hashlib


class Connection(threading.Thread):
    """This class is used to represent a connection for the Node class"""

    def __init__(self, main_node, sock, sock_id, hostname, port):
        super(Connection, self).__init__()
        self.main_node = main_node
        self.hostname = hostname
        self.port = port
        self.sock = sock
        self.id = str(sock_id)

        # End of transmission character for the network streaming messages.
        self.EOT_CHAR = 0x04.to_bytes(1, 'big')

        # Byte encoding format
        self.encoding = 'utf-8'

        self.received_messages = []

        self.sock.settimeout(10.0)

        self.terminate_flag = threading.Event()

        self.main_node.debug_print(
            f'Connection.init: Started with client {self.id} | {self.hostname} : {self.port}')

    def send(self, msg):
        try:
            self.sock.sendall(msg.encode(self.encoding) + self.EOT_CHAR)

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

        buffer = b''

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

                    #TODO: send main thread the message
                    self.main_node.debug_print(self.received_messages.pop())
                    #end todo

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

    def __init__(self, hostname, port, id, max_connections=-1, debug=False):
        super(Node, self).__init__()

        self.hostname = hostname
        self.port = port

        if id == None:
            self.id = self.generate_id()

        else:
            self.id = str(id)  # Make sure the ID is a string!

        # Set of the Connection objects that connect TO this node (connection->this)
        self.inbound_connections = set()
        # Set of the Connections objects this node connects to (this->connection)
        self.outbound_connections = set()
        # in general, items in inboud_connections = items in outbound_connections

        # A list of nodes that should be reconnected to whenever the connection was lost
        self.reconnect_to_nodes = set()

        self.terminate_flag = threading.Event()

        # Message counters to make sure everyone is able to track the total messages
        self.message_count_send = 0
        self.message_count_recv = 0
        self.message_count_rerr = 0

        # Connection limit of inbound nodes (nodes that connect to us)
        self.max_connections = max_connections

        # Debugging on or off!
        self.debug = debug

        self.encoding = 'utf-8'
        self.EOT_CHAR = 0x04.to_bytes(1, 'big')

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
        print(f'[{self.id}] Intitialized node at {self.hostname} : {self.port}')

    def send(self, conn, msg):
        self.message_count_send = self.message_count_send + 1
        if conn in self.inbound_connections or conn in self.outbound_connections:
            conn.send(msg)
        else:
            self.debug_print(
                'Node.send: Could not send the data, node is not found!')
            self.debug_print(conn.id)
            self.debug_print([x.id for x in self.outbound_connections])
            self.debug_print(conn)
            self.debug_print([x for x in self.outbound_connections])

    def broadcast(self, msg, exclude=set()):
        self.message_count_send = self.message_count_send + 1
        for conn in self.inbound_connections.union(self.outbound_connections):
            if conn in exclude:
                self.debug_print(
                    'Node.send: Excluding node in sending the message')
            else:
                self.message_count_send = self.message_count_send + 1
                conn.send(msg)

    def connect_with(self, hostname, port):
        """Connects to the node at given hostname and port"""
        if hostname == self.hostname and port == self.port:
            self.debug_print(
                'Node.connect_with: Cannot connect with yourself!')
            return False

        # Check if node is already connected with this node!
        for conn in self.outbound_connections:
            if conn.port == port and conn.hostname == hostname:
                self.debug_print(f'Node.connect_with: Already connected with node {conn.id}')
                return True

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.debug_print(
                f'Node.connect_with: Connecting with {hostname} : {port}')
            sock.connect((hostname, port))

            # Basic information exchange (not secure) of the id's of the nodes!
            # Send my id and port to the connected node!
            sock.sendall(self.id.encode(self.encoding) + self.EOT_CHAR)

            # When a node is connected, it sends its id!
            connected_node_id = sock.recv(4096)
            eot_pos = connected_node_id.find(self.EOT_CHAR)
            connected_node_id = connected_node_id[:eot_pos].decode(self.encoding)

            # Cannot connect with yourself
            if connected_node_id == self.id:
                self.debug_print(
                    'Node.connect_with: You cannot connect with yourself?!')
                #sock.send('CLOSING: Already having a connection together'.encode(self.encoding))
                sock.close()
                return True

            # Cannot connect if we already have a connection
            for conn in self.inbound_connections:
                #self.debug_print(type(conn))
                if conn.id == connected_node_id:
                    self.debug_print(
                        f'Node.connect_with: Already connected with node {conn.id}')
                    #sock.send('CLOSING: Already having a connection together'.encode(self.encoding))
                    sock.close()
                    return True

            connection_thread = Connection(
                self, sock, connected_node_id, hostname, port)
            connection_thread.start()
            self.outbound_connections.add(connection_thread)
            self.debug_print(
                f'Node.connect_with: Connected with node {connected_node_id}')
            self.debug_print(f'Node.connect_with: Outbound connections {[conn.id for conn in self.outbound_connections]}')
            return True

        except Exception as e:
            self.debug_print(
                f'Node.connect_with: Could not connect with node. {str(e)}')
            return False

    def disconnect_with(self, conn):
        if conn in self.outbound_connections:
            self.debug_print(
                f'Node.disconnect_with: Disconnecting with node {conn.id}')
            conn.stop()
        else:
            self.debug_print(
                f'Node.disconnect_with: Cannot disconnect with node {conn.id}')

    def node_disconnected(self, conn):
        """While the same nodeconnection class is used, the class itself is not able to
           determine if it is a inbound or outbound connection. This function is making
           sure the correct method is used."""
        self.debug_print(f'Node.node_disconnected: {conn.id}')

        if conn in self.inbound_connections:
            self.inbound_connections.remove(conn)
            self.debug_print(
                f'Node.node_disconnected: Disconnected with node {conn.id}')

        if conn in self.outbound_connections:
            self.outbound_connections.remove(conn)
            self.debug_print(
                f'Node.node_disconnected: Disconnected with node {conn.id}')

    def stop(self):
        """This removes all outbound connections, inbound connections and gracefully exits"""
        self.debug_print(f'Node.stop: Stopping node')
        self.terminate_flag.set()

    def run(self):
        while not self.terminate_flag.is_set():  # Check whether the thread needs to be closed
            try:
                self.debug_print('Node.run: Waiting for incoming connection')
                sock, addr = self.sock.accept()

                # When the maximum connections is reached, it disconnects the connection 
                if self.max_connections == -1 or len(self.inbound_connections) < self.max_connections:
                    
                    # Basic information exchange (not secure) of the id's of the nodes!
                    connected_node_id = sock.recv(4096)
                    eot_pos = connected_node_id.find(self.EOT_CHAR)
                    connected_node_id = connected_node_id[:eot_pos].decode(self.encoding)
                    sock.sendall(self.id.encode(self.encoding) + self.EOT_CHAR)
                    
                    connection_thread = Connection(self, sock, connected_node_id, addr[0], addr[1])
                    connection_thread.start()

                    self.inbound_connections.add(connection_thread)
                    self.debug_print(f'Node.run: Connected with node {connected_node_id}')
                    self.debug_print(f'Node.run: Inbound connections {[conn.id for conn in self.inbound_connections]}')
                else:
                    self.debug_print('Node.run: Maximum connection limit reached. New connection closed')
                    sock.close()
            
            except socket.timeout:
                self.debug_print('Node.run: Connection timeout!')

            except Exception as e:
                raise e

            #self.reconnect_nodes()

            time.sleep(0.01)

        self.debug_print('Node.run: Stopping node')
        for conn in self.inbound_connections:
            conn.stop()

        for conn in self.outbound_connections:
            conn.stop()

        time.sleep(1)

        for conn in self.inbound_connections:
            conn.join()

        for conn in self.outbound_connections:
            conn.join()

        self.sock.settimeout(None)   
        self.sock.close()
        print(f'[{self.id}] Stopped node at {self.hostname} : {self.port}')


