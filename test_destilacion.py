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
import keras


#Laberinto 3x3
'''setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        } '''
#Laberinto 4x4
'''setup = { 'width': 4,
        'height': 4,
        'walls': [(1,1),(2,0),(2,1),(3,1),(3,3)],
        'start': (0,0),
        'goal': (2,3),
        'theta': 0
        }'''
#Laberinto 5x5
setup = { 'width': 5,
        'height': 5,
        'walls': [(1,1),(3,0),(2,2),(2,3),(3,1),(4,2)],
        'start': (0,0),
        'goal': (3,2),
        'theta': 0
        }
#Laberinto 6x6
'''setup = { 'width': 6,
        'height': 6,
        'walls': [(1,1),(0,5),(1,2),(1,3),(3,3),(2,4),(2,5),(5,4)],
        'start': (0,0),
        'goal': (5,5),
        'theta': 0
        }'''

env = gym.make('TurtleRobotEnv-v1_2', **setup)


#####LABERINTOS 3X3 Y 4X4
model = Sequential()
#Input is 1 observation vector, and the number of observations in that vector 
model.add(Input(shape=(1,5)))  
model.add(Flatten())
#Hidden layers with 24 nodes each
model.add(Dense(36, activation='relu'))
model.add(Dense(36, activation='relu'))
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

#Laberintos 5x5 y 6x6
'''model = keras.Sequential()
#Input is 1 observation vector, and the number of observations in that vector 
model.add(Input(shape=(1,5)))  
model.add(Flatten())
#Hidden layers with 24 nodes each
model.add(Dense(48, activation='relu'))                             
model.add(Dense(92, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(4, activation='linear')) '''

model=keras.models.load_model('models/5x5distilled_weights_12x2.h5',compile=False)


state=env.reset()
done=False
contador=0


print(model.predict(np.reshape(state,(1,1,5))))
while not done:
    action=model.predict(np.reshape(state,(1,1,5)))
    action=np.argmax(action[0])
    new_state, reward, done, info = env.step(action)
    env.render(action=action, reward=reward)
    state=new_state
    contador+=1
print(contador)


