#!/usr/bin/env python
#coding=utf-8
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import pickle
import random
from turtlebot_communication import tcp_dqn
import copy
import socket
import json


delante=None
derecha=None
izquierda=None
def ScanCallback(msg):
    global delante
    global derecha
    global izquierda
    delante=msg.ranges[0]
    derecha=msg.ranges[1080]
    izquierda=msg.ranges[360]
    #print(len(msg.ranges))

def turn_left():
    move.angular.z=0.79
    move.linear.x=0
    for i in range(7):
        pub.publish(move)
        rate.sleep()

def turn_right():
    move.angular.z=-0.79
    move.linear.x=0
    for i in range(7):
        pub.publish(move)
        rate.sleep()

def move_ahead():
    move.angular.z=0
    move.linear.x=0.22
    for i in range(5):
        pub.publish(move)
        rate.sleep()
def stop():
    move.angular.z=0
    move.linear.x=0
    pub.publish(move)
    rate.sleep()
def actualiza_estado(pos_actual,pos_final,orientacion):
    posi_actual=copy.copy(pos_actual)
    posi_final=copy.copy(pos_final)
    orient=copy.copy(orientacion)
    '''right=1 if derecha<0.4 else 0
    left=1 if izquierda<0.4 else 0
    front=1 if delante<0.4 else 0'''
    right=(orient==0 and posi_actual[1]+1<=max_y and (posi_actual[0],posi_actual[1]+1) not in walls) or (orient==1 and posi_actual[0]+1<=max_x and (posi_actual[0]+1,posi_actual[1]) not in walls) or (orient==2 and posi_actual[1]-1>=0 and (posi_actual[0],posi_actual[1]-1) not in walls) or (orient==3 and posi_actual[0]-1>=0 and (posi_actual[0]-1,posi_actual[1]) not in walls)
    obs_derecha=1 if right==True else 0
    front=(orient==0 and posi_actual[0]-1>=0 and (posi_actual[0]-1,posi_actual[1]) not in walls) or (orient==1 and posi_actual[1]+1<=max_y and (posi_actual[0],posi_actual[1]+1) not in walls) or (orient==2 and posi_actual[0]+1<=max_x and (posi_actual[0]+1,posi_actual[1]) not in walls) or (orient==3 and posi_actual[1]-1>=0 and (posi_actual[0],posi_actual[1]-1) not in walls)
    obs_delante=1 if front==True else 0
    left=(orient==0 and posi_actual[1]-1>=0 and (posi_actual[0],posi_actual[1]-1) not in walls) or (orient==1 and posi_actual[0]-1>=0 and (posi_actual[0]-1,posi_actual[1]) not in walls) or (orient==2 and posi_actual[1]+1<=max_y and (posi_actual[0],posi_actual[1]+1) not in walls) or (orient==3 and posi_actual[0]+1<=max_x and (posi_actual[0]+1,posi_actual[1]) not in walls)
    obs_izquierda=1 if left==True else 0
    rel_goal=0
    if posi_actual[0]<posi_final[0]:  
        rel_goal=10
    elif posi_actual[0]>posi_final[0]:  
        rel_goal=20
    elif posi_actual[0]==posi_final[0]:
        rel_goal=30

    if posi_actual[1]<posi_final[1]:
        rel_goal+=1
    elif posi_actual[1]>posi_final[1]:
        rel_goal+=2
    elif posi_actual[1]==posi_final[1]:
        rel_goal+=3
    distance= abs(posi_actual[0]-posi_final[0])+abs(posi_actual[1]-posi_final[1])
    return [obs_derecha,obs_delante,obs_izquierda,rel_goal,distance]
def choose_action(epsilon,state):
    if np.random.random()<=epsilon:
        return random.randint(0,2)
    else:
        return np.random.choice((np.argwhere(Q[state,:]==np.amax(Q[state,:]))).flatten())

def check_action(pos_actual,orientacion,action):
    posi_actual=copy.copy(pos_actual)
    orient=copy.copy(orientacion)
    
    if action == 1 or action == 2:
        return True
    else:
        if orient==0:
            if posi_actual[0]-1<0 or (posi_actual[0]-1,posi_actual[1]) in walls:
                return False
            else:
                return True
        elif orient==1:
            if posi_actual[1]+1>2 or (posi_actual[0],posi_actual[1]+1) in walls:
                return False
            else:
                return True
        elif orient==2:
            if posi_actual[0]+1>2 or (posi_actual[0]+1,posi_actual[1]) in walls:
                return False
            else:
                return True
        if orient==3: 
            if posi_actual[1]-1<0 or (posi_actual[0],posi_actual[1]-1) in walls:
                return False
            else:
                return True
def actualiza_posicion(posi_actual,orient,action):
    can_execute_action=check_action(posi_actual,orient,action)
    pos=copy.copy(posi_actual)
    ori=copy.copy(orient)
    if can_execute_action:
        if action==0:
            if orient==0:
                pos[0]=pos[0]-1
            elif orient==1:
                pos[1]=pos[1]+1
            elif orient==2:
                pos[0]=pos[0]+1
            elif orient==3:
                pos[1]=pos[1]-1
        if action==1:
            ori=orient-1
            if ori<0:
                ori=3
        if action==2:
            ori=orient+1
            if ori>3:
                ori=0
        return pos,ori
    else:
        return pos,ori
    
def execute_action(action):
    if action==0:
        move_ahead()
    elif action==1:
        turn_left()
    elif action ==2:
        turn_right()




walls=[(1,1),(0,2)]
goal=[1,2]
max_x=2
max_y=2
rospy.init_node('turtle')
sub=rospy.Subscriber('/scan',LaserScan,ScanCallback)
pub=rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=1)
rate=rospy.Rate(2)
move=Twist()

while derecha==None or izquierda==None or delante==None:
    rate.sleep()
list_acciones=[]




host = '192.168.0.101'
port = 5000


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(5)
print('Servidor escuchando')
orientacion=0
goal=[1,2]
pos_inicial=[0,0]
pos_actual=[0,0]
state=actualiza_estado(pos_actual,goal,orientacion)
while True:
    connection, client_address = server_socket.accept()
    print('Cliente conectado:', client_address)

    try:
       
        while True:
            data = connection.recv(1024)
            if data:
                mensaje=data.decode()
                mensaje=json.loads(mensaje)
                mensaje[0]=mensaje[0].encode('utf-8')
                print('Mensaje recibido:', mensaje[0],type(mensaje[0]))
                if mensaje[0]=='reset':
                    state=actualiza_estado(pos_inicial,goal,0)
                    response=json.dumps(state)
                    connection.sendall(response.encode())
                    print('reset')
                if mensaje[0]=='step':
                    pos_actual,orientacion=actualiza_posicion(mensaje[2],mensaje[3],mensaje[1])
                    state=actualiza_estado(mensaje[2],goal,mensaje[3])
		    message=[state,pos_actual,orientacion]
                    response=json.dumps(message)
                    
		    can_execute_action=check_action(mensaje[2],mensaje[3],mensaje[1])
		    if can_execute_action:
			execute_action(mensaje[1])
			stop()
		    connection.sendall(response.encode())
                    print('step',mensaje[2],mensaje[3])
                if mensaje[0]=='get_sensor_readings':
                    response=json.dumps(state)
                    connection.sendall(response.encode())
                    print('enviado')
                if mensaje[0]=='reward':
                    if mensaje[1]==goal:
                        reward=int(10)
                        done=True
                    else:
                        reward=int(-1)
                        done=False
                    response=json.dumps([reward,done])
                    print('reward')
                    connection.sendall(response.encode())
            else:
                break

    except:
	    connection.close()
        
