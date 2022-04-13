from pydoc import cli
import socket
import sys
from time import sleep

class Client:
    def __init__(self, header=128, format='utf-8', disconnect_msg='Disconnecting'):
        self.header = header
        self.format = format
        self.disconnect_msg = disconnect_msg
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


    def connect(self, hostname, port):
        self._s.connect((hostname, port))
        print(f"[connect] Client bound to hostname {hostname}, port {port}")


    def send(self, msg):
        m = msg.encode(self.format)
        m_length = len(m)
        m_length = str(m_length).encode(self.format)
        m_length += b' ' * (self.header - len(m_length))
        self._s.send(m_length)
        self._s.send(m)
        print(f"[send] Sent {msg}")

    def disconnect(self):
        self.send(self.disconnect_msg)
        self._s.close()
        print(f"[disconnect] Disconnected with server")


if __name__ == '__main__': 
    if len(sys.argv) == 3:
        hostname = sys.argv[1]
        port = sys.argv[2]
    else:
        hostname = socket.gethostbyname(socket.gethostname())
        port = 5050

    client = Client() 
    client.connect(hostname, port)
    client.send("Hello World!")
    input()
    client.send("Goodbye!")
    input()
    client.disconnect()