import socket
import json


host = '192.168.0.102'
port = 5000


def server(dqn):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print('Servidor escuchando')

    while True:
        connection, client_address = server_socket.accept()
        print('Cliente conectado:', client_address)

        try:
            while True:
                data = connection.recv(1024)
                if data:
                    mensaje=data.decode()
                    mensaje=json.loads(mensaje)
                    print('Mensaje recibido:', mensaje)
                    response=str(dqn.forward(mensaje))
                    print(response,type(response))
                    connection.sendall(response.encode())
                else:
                    break

        finally:
            connection.close()
