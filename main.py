import random
import gym
import turtle_robot_gym
import numpy as np
import matplotlib.pyplot as plt

# create the turtle environment

#Laberinto 3x3
'''setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        }''' 
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
setup = { 'width': 6,
        'height': 6,
        'walls': [(1,1),(0,5),(1,2),(1,3),(3,3),(2,4),(2,5),(5,4)],
        'start': (0,0),
        'goal': (5,5),
        'theta': 0
        }

env = gym.make('TurtleRobotEnv-v1', **setup)


def choose_action(epsilon,state):
    if np.random.random() <= epsilon:
        return random.randint(0,2)
    else:
        return np.random.choice((np.argwhere(Q[state,:] == np.amax(Q[state,:]))).flatten())

for i in range(500):
    epsilon=1.0
    Q = np.zeros([288, 3])
    lr = 0.15
    y = 0.99
    eps = 500
    visited_states=[]
    list_acciones=[]
    print(i)
    for i in range(eps):
        # initialize the environment
        s=env.reset()
        OldStrState=''.join(s)
        if OldStrState not in visited_states: visited_states.append(OldStrState)
        OldState=visited_states.index(OldStrState)
        done = False
        n_acciones=0
        while not done:  

            # choose a random action
            #action = random.randint(0, 2)
            action=choose_action(epsilon,OldState)
            
            # take the action and get the information from the environment
            new_state, reward, done, info = env.step(action)

            StrState=''.join(new_state)
            if StrState not in visited_states: visited_states.append(StrState)
            NewState=visited_states.index(StrState)

            #Update Q table
            Q[OldState,action] = Q[OldState,action] + lr*(reward + y*np.max(Q[NewState,:]) - Q[OldState,action])

            OldState=NewState
            n_acciones+=1
            # show the current position and reward
            #env.render(action=action, reward=reward)
        list_acciones.append(n_acciones)
        
        if done:
            epsilon=max(epsilon*0.99,0.01)

    f=open('data/AccionesQLeaning.txt','a')
    f.write(str(list_acciones))
    f.write('\n')
    f.close()

'''x=[i for i in range(len(list_acciones))]

plt.plot(x,list_acciones)
plt.grid()
plt.show()'''

print(Q[:len(visited_states),:])


#Using Q table obtained after all episodes
s=env.reset()
OldStrState=''.join(s)
if OldStrState not in visited_states: visited_states.append(OldStrState)
OldState=visited_states.index(OldStrState)

done=False
while not done:

    action = np.random.choice((np.argwhere(Q[OldState,:] == np.amax(Q[OldState,:]))).flatten())
    new_state, reward, done, info = env.step(action)

    StrState=''.join(new_state)
    if StrState not in visited_states: visited_states.append(StrState)
    NewState=visited_states.index(StrState)
    OldState=NewState
    env.render(action=action, reward=reward)
