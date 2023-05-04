import gym
import turtle_robot_gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
from tensorflow.keras.models import load_model
from keras import Sequential
from keras.layers import Input, Flatten, Dense
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent


#Laberinto 3x3
'''setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        } '''
#Laberinto 4x4
setup = { 'width': 4,
        'height': 4,
        'walls': [(1,1),(2,0),(2,1),(3,1),(3,3)],
        'start': (0,0),
        'goal': (2,3),
        'theta': 0
        }
#Laberinto 5x5
'''setup = { 'width': 5,
        'height': 5,
        'walls': [(1,1),(3,0),(2,2),(2,3),(3,1),(4,2)],
        'start': (0,0),
        'goal': (3,2),
        'theta': 0
        } '''  
#Laberinto 6x6
'''setup = { 'width': 6,
        'height': 6,
        'walls': [(1,1),(0,5),(1,2),(1,3),(3,3),(2,4),(2,5),(5,4)],
        'start': (0,0),
        'goal': (5,5),
        'theta': 0
        }'''

env = gym.make('TurtleRobotEnv-v1_2', **setup)

#####LABERINTOS 3X3 Y 4X4  de 3 CAPAS DE 64
model = Sequential()
#Input is 1 observation vector, and the number of observations in that vector 
model.add(Input(shape=(1,5)))  
model.add(Flatten())
#Hidden layers with 24 nodes each
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
#Output is the number of actions in the action space
model.add(Dense(env.action_space.n, activation='linear'))
#####LABERINTOS 3X3 Y 4X4
'''model = Sequential()
#Input is 1 observation vector, and the number of observations in that vector 
model.add(Input(shape=(1,5)))  
model.add(Flatten())
#Hidden layers with 24 nodes each
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
#Output is the number of actions in the action space
model.add(Dense(env.action_space.n, activation='linear'))'''

#######LABERINTO 5X5 Y 6X6
'''model = Sequential()
#Input is 1 observation vector, and the number of observations in that vector 
model.add(Input(shape=(1,5)))  
model.add(Flatten())
#Hidden layers with 24 nodes each
model.add(Dense(96, activation='relu'))
model.add(Dense(192, activation='relu'))
model.add(Dense(96, activation='relu'))
#Output is the number of actions in the action space
model.add(Dense(env.action_space.n, activation='linear')) '''

memory = SequentialMemory(limit=50000, window_length=1)

# setup the Linear annealed policy with the EpsGreedyQPolicy as the inner policy
policy =  LinearAnnealedPolicy(inner_policy=  EpsGreedyQPolicy(),   # policy used to select actions
                               attr='eps',                          # attribute in the inner policy to vary             
                               value_max=1.0,                       # maximum value of attribute that is varying
                               value_min=0.1,                       # minimum value of attribute that is varying
                               value_test=0.05,                     # test if the value selected is < 0.05
                               nb_steps=10000)  
dqn = DQNAgent(model=model,                     # Q-Network model
               nb_actions=env.action_space.n,   # number of actions
               memory=memory,                   # experience replay memory
               nb_steps_warmup=25,              # how many steps are waited before starting experience replay
               target_model_update=1e-2,        # how often the target network is updated
               policy=policy) 
dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae','accuracy'])

dqn.load_weights('models/4x4_turtle_weights64x3.h5')



f1=open('data/estados.txt','w')
f2=open('data/predicciones.txt','w')
contador=0
for i in range(500):
    state=env.reset()
    done=False
    print('ESTADO',state)

    while not done:
        action=dqn.forward(state)
        predicciones=dqn.compute_q_values(np.reshape(state,(1,5)))
        f1.write(str(state[0])+' '+str(state[1])+' '+str(state[2])+' '+str(state[3])+' '+str(state[4])+' '+'\n')
        #print(np.reshape(state,(1,5)))
        #print(predicciones)
        f2.write(str(predicciones[0])+' '+str(predicciones[1])+' '+str(predicciones[2])+' '+str(predicciones[3])+'\n')
        #print(type(predicciones))
        new_state, reward, done, info = env.step(action)
        #env.render(action=action, reward=reward)
        state=new_state
        contador+=1
print(contador)
f1.close()
f2.close()

