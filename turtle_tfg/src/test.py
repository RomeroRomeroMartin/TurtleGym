#!/usr/bin/env python
#coding=utf-8
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import pickle
import random
import copy

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
    move.angular.z=0.78
    move.linear.x=0
    for i in range(7):
        pub.publish(move)
        rate.sleep()

def turn_right():
    move.angular.z=-0.78
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
    
def execute_action(action):
    if action==0:
        move_ahead()
    elif action==1:
        turn_left()
    elif action ==2:
        turn_right()
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

with open('/home/ap/grupo_primero_0/catkin_ws/src/turtle_tfg/models/Qlear3x3_1.pkl','rb') as f:
    data=pickle.load(f)
 
Q,visited_states=data

list_acciones=[]


orientacion=0
s=actualiza_estado((0,0),goal,orientacion)
s=list(map(str,s))
OldStrState=''.join(s)
if OldStrState not in visited_states: visited_states.append(OldStrState)
OldState=visited_states.index(OldStrState)
done=False
n_acciones=0
pos_actual=[0,0]
while not done:
    action=np.random.choice((np.argwhere(Q[OldState,:]==np.amax(Q[OldState,:]))).flatten())

    can_execute_action=check_action(pos_actual,orientacion,action)

    if can_execute_action:
        execute_action(action)
        stop()
        pass

    new_pos,new_orientacion=actualiza_posicion(pos_actual,orientacion,action)

    new_state=actualiza_estado(new_pos,goal,new_orientacion)
    if new_pos==goal:
        reward=10
        done=True
    else:
        reward=-1

    new_state=list(map(str,new_state))
    StrState=''.join(new_state)
    if StrState not in visited_states: visited_states.append(StrState)
    NewState=visited_states.index(StrState)

    OldState=NewState
    pos_actual=new_pos
    orientacion=new_orientacion
    print(n_acciones)
    print(pos_actual,orientacion)
    n_acciones+=1
    list_acciones.append(n_acciones)






        

#while not rospy.is_shutdown():


    #print('Izquierda: ',izquierda,'Delante: ',delante,'Derecha: ',derecha)
    #rate.sleep()	       
#move_ahead()
#turn_right()
#move_ahead()
#rate.sleep()
#turn_right()
#move_ahead()
#turn_left()
#stop()
#rate.sleep()

        
