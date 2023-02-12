import gym
import math
import turtle_robot_gym
import keras
import random
import numpy as np
from collections import deque
from  keras.models import  Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
max_score = 0

n_episodes = 200
n_win_tick = 1000
max_env_steps = 1000

gamma = 1.0
epsilon = 1.0 #exploration
epsilon_min = 0.01
epsilon_decay = 0.97

alpha = 0.01 # learning rate
alpha_decay = 0.01
alpha_test_factor = 1.0

batch_size = 288
monitor = False
quiet = False

#environment Parameters
memory = deque(maxlen=10000)
setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        } 

env = gym.make('TurtleRobotEnv-v1_2', **setup)

if max_env_steps is not None:
    env._max_episode_steps = max_env_steps

with tf.device("/GPU:0"):

    model = Sequential()
    model.add(Dense(24,input_dim=5,activation='relu'))
    model.add(Dense(24, activation='relu'))
    #model.add(Dense(96, activation='relu'))
    #model.add(Dense(48, activation='relu'))
    #model.add(Dense(24, activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=alpha))


    target_model = Sequential()
    target_model.add(Dense(24,input_dim=5,activation='relu'))
    target_model.add(Dense(24, activation='relu'))
    #target_model.add(Dense(96, activation='relu'))
    #target_model.add(Dense(48, activation='relu'))
    #target_model.add(Dense(24, activation='relu'))
    target_model.add(Dense(4,activation='relu'))
    target_model.compile(loss='mse', optimizer=Adam(learning_rate=alpha, weight_decay=alpha_decay))
#Define functions
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
def choose_action(state, epsilon):
    if np.random.random() <= epsilon:
        return random.randint(0,2)
    else:
        lst=model.predict(state,verbose=0)
        flat_arr = lst.flatten()
        max_value = np.max(flat_arr)
        max_indices = np.argwhere(flat_arr == max_value)
        max_index = tuple(random.choice(max_indices).tolist())
        #return np.random.choice((np.argwhere(model.predict(state,verbose=0) == np.amax(model.predict(state,verbose=0)))).flatten())
        #return np.argmax(model.predict(state,verbose=0))
        return max_index[0]

def preprocess(state):
    return np.reshape(state, [1,5])
def replay(batch_size,epsilon):
    with tf.device("/GPU:0"):
        x_batch, y_batch = [], []
    minibatch = random.sample(memory, min(len(memory), batch_size))
    for state, action, reward, next_state, done in minibatch:
        y_target = target_model.predict(state,verbose=0)
        y_target_next=target_model.predict(next_state,verbose=0)
        flat_arr = y_target_next.flatten()
        max_value = np.max(flat_arr)
        max_indices = np.argwhere(flat_arr == max_value)
        max_index = tuple(random.choice(max_indices).tolist())
        #y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state,verbose=0)[0])
        y_target[0][action] = reward if done else reward + gamma * max_value
        with tf.device("/GPU:0"):
            x_batch.append(state[0])
            y_batch.append(y_target[0])
    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
# run function
def run():
    global max_score
    scores = deque(maxlen=100)
    epsilon=1.0
    for e in range(n_episodes):

        state = preprocess(env.reset())
        done = False
        i = 0
        while not done:
            action = choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            #env.render(action=action,reward=reward)
            next_state = preprocess(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            if i >500:
                break
            i += 1
        print('Finaliza episodio: ',e,' ticks: ',i)
        f=open('ticks.txt','a')
        f.write(str(i)+',')
        f.close()
        f2=open('epsilon.txt','a')
        f2.write(str(epsilon)+',')
        f2.close()
        if done:
            epsilon = max(epsilon * 0.92, 0.01)


        replay(batch_size, epsilon)

        if e%6==0:
            target_model.set_weights(model.get_weights())

        if e==50 or e==100 or e==150:
            model.save(str(e)+'model_weights.h5')

    model.save('model_weights.h5')
    return e

#Training the network

run()

print("Finished")