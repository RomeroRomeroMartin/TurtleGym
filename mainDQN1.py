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

max_score = 0

n_episodes = 5000
n_win_tick = 1000
max_env_steps = 1000

gamma = 1.0
epsilon = 1.0 #exploration
epsilon_min = 0.01
epsilon_decay = 0.999

alpha = 0.01 # learning rate
alpha_decay = 0.01
alpha_test_factor = 1.0

batch_size = 288
monitor = False
quiet = False

#environment Parameters
memory = deque(maxlen=100000)
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




# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             break


model = Sequential()
model.add(Dense(24,input_dim=5,activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(4,activation='relu'))
model.compile(loss='mse', optimizer=Adam(learning_rate=alpha, weight_decay=alpha_decay))

#Define functions
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
def choose_action(state, epsilon):
    if np.random.random() <= epsilon:
        return random.randint(0,2)
    else:
        return np.argmax(model.predict(state))
def get_epsilon(t):
    return max(epsilon_min, min(epsilon,1.0 - math.log10((t+1)*epsilon_decay)))
def preprocess(state):
    return np.reshape(state, [1,5])
def replay(batch_size,epsilon):
    x_batch, y_batch = [], []
    minibatch = random.sample(memory, min(len(memory), batch_size))
    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])
    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
# run function
def run():
    global max_score
    scores = deque(maxlen=100)
    visited_states=[]
    epsilon=1.0
    for e in range(n_episodes):
        '''if e > n_episodes-2:
            global epsilon
            epsilon = 0.0'''

        state = preprocess(env.reset())
        done = False
        i = 0
        while not done:
            action = choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            env.render()
            next_state = preprocess(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1
        if i > max_score:
            max_score = i
            # Save the weights
            model.save(str(max_score) + 'model_weights.h5')
        if done:
            epsilon = max(epsilon * 0.99, 0.01)
            '''# Save the model architecture
            with open(str(max_score) + 'model_architecture.json', 'w') as f:
                f.write(model.to_json())'''


        scores.append(i)
        mean_score = np.mean(scores)
        if mean_score >= n_win_tick and e >= 100:
            if not quiet: print("Ran " + str(e) + " episodes. Solved after " + str(e-100) + "trials")
            # Save the weights
            model.save(str(max_score) + 'final_model_weights.h5')

            # Save the model architecture
            '''with open(str(max_score) + 'final_model_architecture.json', 'w') as f:
                f.write(model.to_json())'''

            return e-100
        if e % 100 == 0 and not quiet:
            print("episode " + str(e) + " mean survival time over last 100 episodes was " + str(mean_score) + " ticks")
        replay(batch_size, epsilon)
    if not quiet:
        print("did not solve after " + str(e) + " episodes")
    return e

#Training the network

run()

print("max achived score is : " + str(max_score))