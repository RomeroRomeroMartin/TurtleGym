import gym
#import matplotlib.pyplot as plt
import turtle_robot_gym
from keras import Sequential
from keras.layers import Input, Flatten, Dense
import rl
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
import tensorflow as tf
#from tensorflow.keras.optimizers import Adam
laberinto='3x3'

if laberinto=='3x3':
    #Laberinto 3x3
    setup = { 'width': 3,
            'height': 3,
            'walls': [(1,1),(0,2)],
            'start': (0,0),
            'goal': (1,2),
            'theta': 0
            } 
if laberinto=='4x4':
    #Laberinto 4x4
    setup = { 'width': 4,
            'height': 4,
            'walls': [(1,1),(2,0),(2,1),(3,1),(3,3)],
            'start': (0,0),
            'goal': (2,3),
            'theta': 0
            } 
if laberinto=='5x5':
    #Laberinto 5x5
    setup = { 'width': 5,
            'height': 5,
            'walls': [(1,1),(0,3),(2,2),(2,3),(3,1),(4,2)],
            'start': (0,0),
            'goal': (3,2),
            'theta': 0
            }
if laberinto=='6x6':
    #Laberinto 6x6
    setup = { 'width': 6,
            'height': 6,
            'walls': [(1,1),(0,5),(1,2),(1,3),(3,3),(2,4),(2,5),(5,4)],
            'start': (0,0),
            'goal': (5,5),
            'theta': 0
            }

ENV_NAME = 'TurtleRobotEnv-v1_2'
env = gym.make(ENV_NAME, **setup)

print('Observation space ', env.observation_space.shape[0])
print('Action space ', env.action_space.n)


# setup experience replay buffer
memory = SequentialMemory(limit=50000, window_length=1)

# setup the Linear annealed policy with the EpsGreedyQPolicy as the inner policy
policy =  LinearAnnealedPolicy(inner_policy=  EpsGreedyQPolicy(),   # policy used to select actions
                attr='eps',                          # attribute in the inner policy to vary             
                value_max=1.0,                       # maximum value of attribute that is varying
                value_min=0.1,                       # minimum value of attribute that is varying
                value_test=0.05,                     # test if the value selected is < 0.05
                nb_steps=10000)                      # the number of steps between value_max and value_min

if laberinto=='5x5' or laberinto=='6x6':
    #Feed-Forward Neural Network Model for Deep Q Learning (DQN)
    model = Sequential()
    #Input is 1 observation vector, and the number of observations in that vector 
    model.add(Input(shape=(1,5)))  
    model.add(Flatten())
    #Hidden layers with 24 nodes each
    model.add(Dense(96, activation='relu'))                             
    model.add(Dense(192, activation='relu'))
    model.add(Dense(96, activation='relu'))    
    model.add(Dense(env.action_space.n, activation='linear')) 
if laberinto=='3x3' or laberinto=='4x4':
    #Feed-Forward Neural Network Model for Deep Q Learning (DQN)
    model = Sequential()
    #Input is 1 observation vector, and the number of observations in that vector 
    model.add(Input(shape=(1,5)))  
    model.add(Flatten())
    #Hidden layers with 24 nodes each
    model.add(Dense(64, activation='relu'))                             
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(64, activation='relu'))    
    model.add(Dense(env.action_space.n, activation='linear')) 

for i in range(1):
    #Feed-Forward Neural Network Architecture Summary
    print(model.summary())
    #Defining DQN Agent for DQN Model
    dqn = DQNAgent(model=model,                     # Q-Network model
        nb_actions=env.action_space.n,   # number of actions
        memory=memory,                   # experience replay memory
        nb_steps_warmup=25,              # how many steps are waited before starting experience replay
        target_model_update=1e-2,        # how often the target network is updated
        policy=policy)                   # the action selection policy

    # Finally, we configure and compile our agent. 
    #We can use built-in tensorflow.keras Adam optimizer and evaluation metrics            

    dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae','accuracy'])
    #Finally fit and train the agent
    history = dqn.fit(env,nb_steps=25000, visualize=False, verbose=1)

    f=open('data/AccionesDQN.txt','a')
    #f.write(str(history.history['nb_episode_steps']))
    #f.write('\n')
    #f.close()
    # Finally, evaluate and test our algorithm for 20 episodes.
    dqn.test(env, nb_episodes=20, visualize=False)

    # Save weights
    model.save_weights('models/'+laberinto+'_turtle_weights64x3_2.h5')
