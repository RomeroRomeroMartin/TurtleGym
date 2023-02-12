import gym
import turtle_robot_gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
from tensorflow.keras.models import load_model

def preprocess(state):
    return np.reshape(state, [1,5])

def one_hot_encode(value, values):
    # create a list of zeros with the same length as the list of values
    one_hot = [0] * len(values)
    # set the element at the index of the value to 1
    if value in values:
        one_hot[values.index(value)] = 1
    return one_hot
# define a function to preprocess the state
def preprocess_state(state):
    # encode the first, second, and third components as integers
    encoded_state = [int(state[0]), int(state[1]), int(state[2])]
    # encode the fourth component as a one-hot vector
    encoded_state += one_hot_encode(str(state[3]), ['11', '12', '13', '21', '22', '23', '31', '32', '33'])    # encode the last component as a float
    encoded_state += [float(state[4])]
    return encoded_state
def loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        } 

env = gym.make('TurtleRobotEnv-v1_2', **setup)

state=env.reset()
model=load_model('100model_weights.h5')
done=False
while not done:
    state=preprocess(state)
    action=model.predict(state)
    action=np.random.choice((np.argwhere(action == np.amax(action))).flatten())
    new_state, reward, done, info = env.step(action)
    env.render(action=action, reward=reward)
    state=new_state
