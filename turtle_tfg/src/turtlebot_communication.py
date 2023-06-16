import socket
import sys
import json

host= '192.168.0.102'
port=5000


def tcp_dqn(mensaje):
    client_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
    try:

        client_socket.connect((host,port))

        mensaje_turtlebot=json.dumps(mensaje)

        client_socket.sendall(mensaje_turtlebot.encode())

        data=client_socket.recv(1024)

        #print("Mensaje recibido de maquina virtual: ",data.decode())
	return data.decode()

    finally:
        client_socket.close()

if __name__=='__main__':
    tcp_dqn([1,1,1,11,3])
