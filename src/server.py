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

    def bind(self, hostname, port):
        self._s.bind((hostname, port))
        print(f"[bind] Server bound to hostname {hostname}, port {port}")

    def send(self, s, msg):
        msg = msg.encode(self.format)
        msg_length = len(msg)
        msg_length = str(msg_length).encode(self.format)
        msg_length += b' ' * (self.header - len(msg_length))
        s.send(msg_length)
        s.send(msg)

    def broadcast(self):
        pass

    def receive(self, s):
        msg_length = s.recv(self.header).decode(self.format)
        if msg_length:
            msg_length = int(msg_length)
            msg = s.recv(msg_length).decode(self.format)
        return msg

    def handle_client(self, conn, addr):
        print(f"[{addr}] New connection")

        connected = True
        while connected:
            msg = self.receive(conn)

            print(f"[{addr}] {msg}")

            if msg == self.disconnect_msg:
                connected = False

        conn.close()
        print(f"[{addr}] Disconnected")

    def start(self):
        print(f"[start] Sever started")
        self._s.listen()
        clients = 0
        while clients < self.n_nodes:
            # when a new connection occurs, accept and start a thread to handle that client
            conn, addr = self._s.accept()
            thread = threading.Thread(
                target=self.handle_client, args=(conn, addr))
            thread.start()
            clients += 1
            print(f"[start] Active connections: {threading.activeCount() - 1}")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        hostname = sys.argv[1]
        port = sys.argv[2]
    else:
        hostname = socket.gethostbyname(socket.gethostname())
        port = 5050

    server = Server()
    server.bind(hostname, port)
    server.start()
