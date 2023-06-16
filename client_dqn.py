import socket
import sys
import time
import json

host= '192.168.0.101'
port=5000


def tcp_dqn(mensaje):
    client_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
    try:

        client_socket.connect((host,port))

        mensaje_turtlebot=json.dumps(mensaje)

        client_socket.sendall(mensaje_turtlebot.encode())
        datos=client_socket.recv(1024)
        data=datos
        #while data==datos:
        #    data=client_socket.recv(1024)
        
        #print(data)
        data=json.loads(data)
        #print(data)
        return data
        #print("Mensaje recibido de maquina virtual: ",data.decode())
	    #return data.decode()

    finally:
        client_socket.close()
#data=tcp_dqn(['step',0])
#print(data,type(data),type(data[0]))