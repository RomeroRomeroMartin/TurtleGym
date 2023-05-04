import gym
import turtle_robot_gym
import numpy as np
import random
import pickle


#Laberinto 3x3
setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        } 
#Laberinto 4x4
'''setup = { 'width': 4,
        'height': 4,
        'walls': [(1,1),(2,0),(2,1),(3,1),(3,3)],
        'start': (0,0),
        'goal': (2,3),
        'theta': 0
        } '''
#Laberinto 5x5
'''setup = { 'width': 5,
        'height': 5,
        'walls': [(1,1),(3,0),(2,2),(2,3),(3,1),(4,2)],
        'start': (0,0),
        'goal': (3,2),
        'theta': 0
        }''' 
#Laberinto 6x6
'''setup = { 'width': 6,
        'height': 6,
        'walls': [(1,1),(0,5),(1,2),(1,3),(3,3),(2,4),(2,5),(5,4)],
        'start': (0,0),
        'goal': (5,5),
        'theta': 0
        }'''

env = gym.make('TurtleRobotEnv-v1_2', **setup)

with open('data/Qlear3x3.pkl', 'rb') as f:
    data = pickle.load(f)
Q, visited_states = data
print(Q)
print(visited_states)

s=env.reset()
s = list(map(str, s))
OldStrState=''.join(s)
if OldStrState not in visited_states: visited_states.append(OldStrState)
OldState=visited_states.index(OldStrState)
print('STATEEEE',Q[OldState])
done=False
while not done:

    action = np.random.choice((np.argwhere(Q[OldState,:] == np.amax(Q[OldState,:]))).flatten())
    new_state, reward, done, info = env.step(action)
    new_state = list(map(str, new_state))
    StrState=''.join(new_state)
    if StrState not in visited_states: visited_states.append(StrState)
    NewState=visited_states.index(StrState)
    OldState=NewState
    env.render(action=action, reward=reward)