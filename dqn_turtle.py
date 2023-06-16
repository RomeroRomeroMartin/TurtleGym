import socket
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
setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        } 
env = gym.make('TurtleRobotEnv-v1_4', **setup)

#####LABERINTOS 3X3 Y 4X4  de 3 CAPAS DE 64
model = Sequential()
#Input is 1 observation vector, and the number of observations in that vector 
model.add(Input(shape=(1,5)))  
model.add(Flatten())
#Hidden layers with 24 nodes each
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
#Output is the number of actions in the action space
model.add(Dense(env.action_space.n, activation='linear'))

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

#Finally fit and train the agent
history = dqn.fit(env,nb_steps=25000, visualize=False, verbose=1)

#f=open('data/AccionesDQN.txt','a')
#f.write(str(history.history['nb_episode_steps']))
#f.write('\n')
#f.close()
# Finally, evaluate and test our algorithm for 20 episodes.
dqn.test(env, nb_episodes=20, visualize=False)

# Save weights
model.save_weights('models/3x3_RealTurtle_weights.h5')





