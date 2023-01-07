import gym
import turtle_robot_gym
import numpy as np
import tensorflow as tf
import random
from collections import deque


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

# define a function to select an action
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        # select a random action
        action = random.randint(0, 2)
    else:
        # select the action with the highest Q-value
        state = np.expand_dims(state, axis=0)  # add a second dimension to the state array
        q_values = model.predict(state)
        action = np.argmax(q_values)
    return action

# define a function to update the target model
def update_target_model(model, target_model):
    target_model.set_weights(model.get_weights())

# define a function to sample a batch of transitions from the replay buffer
def sample_batch(batch_size):
    # make sure the batch size is a positive integer and less than or equal to the number of transitions in the replay buffer
    batch_size = min(len(replay_buffer), max(0, int(batch_size)))
    transitions = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        } 

env = gym.make('TurtleRobotEnv-v1_2', **setup)

# get the size of the action space and state space
action_size = 3 #env.action_space.n
state_size = 13#env.observation_space.shape[0]

# define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(state_size,), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])

# define the target model
target_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(state_size,), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])

# define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# define the loss function
def loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# define the replay buffer as a deque
replay_buffer = deque()

# define the train function
def train(env, replay_buffer, batch_size, epsilon, discount_factor):
    # sample a batch of transitions from the replay buffer
    states, actions, rewards, next_states, dones = sample_batch(batch_size)

    # preprocess the states and next states
    states = np.vstack([preprocess_state(s) for s in states])
    next_states = np.vstack([preprocess_state(s) for s in next_states])
    # get the Q-values for the next states
    next_q_values = target_model.predict(next_states)

    # set the Q-values for the terminal states to 0
    next_q_values[dones] = 0

    # set the Q-values for the current states
    y = rewards + discount_factor * np.max(next_q_values, axis=1)

    # compile the model
    model.compile(optimizer=optimizer, loss='mse')
    # fit the model on the batch
    model.fit(states, y, batch_size=batch_size, epochs=1,verbose=2)

# define the update_replay_buffer function
def update_replay_buffer(replay_buffer, transitions):
    replay_buffer.extend(transitions)


# define the main loop
def main():
    # set the initial epsilon
    epsilon = 1.0

    # set the discount factor
    discount_factor = 0.95

    # set the batch size
    batch_size = 64

    # set the number of episodes
    n_episodes = 50

    for episode in range(n_episodes):

        # reset the environment
        state = env.reset()

        # preprocess the state
        state = preprocess_state(state)

        # set the initial time step
        t = 0

        while True:
            # select an action
            action = select_action(state, epsilon)

            # take a step in the environment
            next_state, reward, done, _ = env.step(action)

            # preprocess the next state
            next_state = preprocess_state(next_state)
            

            # check if the episode is done
            if done:
                # update the target model
                update_target_model(model, target_model)
                # train the model
                train(env, replay_buffer, batch_size, epsilon, discount_factor)
                # update the epsilon
                epsilon = max(epsilon * 0.99, 0.01)
                # print the results
                print("Episode: {}, Time step: {}, Epsilon: {:.2f}, Reward: {}".format(episode+1, t, epsilon, reward))
                break

            # add the transition to the replay buffer
            update_replay_buffer(replay_buffer, [(state, action, reward, next_state, done)])

            # update the state
            state = next_state

            # update the time step
            t += 1
    # save the model
    model.save('model.h5')

# run the main loop
main()

