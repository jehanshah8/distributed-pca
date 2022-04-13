from email import message
from re import T
import socket
import threading
import sys

from matplotlib.pyplot import connect


class Server:
    def __init__(self, header=128, format='utf-8', disconnect_msg='Disconnecting'):
        self.header = header
        self.format = format
        self.disconnect_msg = disconnect_msg
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

    def handle_client(self, conn, addr):
        print(f"[{addr}] New connection")

        connected = True
        while connected: 
            msg_length = conn.recv(self.header).decode(self.format)
            if msg_length:
                msg_length = int(msg_length)
                msg = conn.recv(msg_length).decode(self.format)
                if msg == self.disconnect_msg:
                    connected = False 
                
                print(f"[{addr}] {msg}")
                #conn.send("received".encode(self.format))
                
                if connected:
                    self.send(conn, "received")
                
        conn.close()
        print(f"[{addr}] Disconnected")

    def start(self):
        print(f"[start] Sever started")
        self._s.listen()
        while True:
            #when a new connection occurs, accept and start a thread to handle that client
            conn, addr = self._s.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()
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
