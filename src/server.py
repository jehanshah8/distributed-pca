from calendar import c
import socket
import threading
import sys
import time


class Server:
    def __init__(self, header=128, format='utf-8', disconnect_msg='Disconnecting', n_nodes=1):
        self.header = header
        self.format = format
        self.disconnect_msg = disconnect_msg
        self.n_nodes = n_nodes
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connections = []

    def bind(self, hostname, port):
        self._s.bind((hostname, port))
        print(f"[bind] Server bound to hostname {hostname}, port {port}")

    def send(self, conn, msg, encoder=None):
        msg = msg.encode(self.format)
        msg_length = len(msg)
        msg_length = str(msg_length).encode(self.format)
        msg_length += b' ' * (self.header - len(msg_length))
        conn.send(msg_length)
        conn.send(msg)

    def broadcast(self, msg, encoder=None):
        [self.send(conn, msg, encoder) for conn in self.connections]

    def receive(self, conn):
        msg_length = conn.recv(self.header).decode(self.format)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(self.format)
        return msg

    def handle_client(self, conn, addr):
        print(f"[{addr}] New connection")

        connected = True
        while connected:
            msg = self.receive(conn)

            print(f"[{addr}] {msg}")

            if msg == self.disconnect_msg:
                connected = False
            else:
                self.send(f"Received")

        conn.close()
        self.connections.remove(conn)
        print(f"[{addr}] Disconnected")

    def start(self):
        print(f"[start] Sever started")
        self._s.listen()
        clients = 0
        while True:
            # when a new connection occurs, accept and start a thread to handle that client
            conn, addr = self._s.accept()
            self.connections.append(conn)
            thread = threading.Thread(
                target=self.handle_client, args=(conn, addr))
            thread.start()
            clients += 1
            print(f"[start] Active connections: {threading.activeCount() - 1}")


if __name__ == '__main__':
    if len(sys.argv) == 2:
        #hostname = sys.argv[1]
        port = sys.argv[1]
    else:
        #hostname = socket.gethostbyname(socket.gethostname())
        port = 5007

    hostname = socket.gethostbyname(socket.gethostname())
    
    server = Server()
    server.bind(hostname, port)
    server.start()
    server.broadcast("broadcasting")
