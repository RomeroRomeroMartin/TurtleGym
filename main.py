import random
import gym
import turtle_robot_gym
import numpy as np
import matplotlib.pyplot as plt

# create the turtle environment

setup = { 'width': 3,
        'height': 3,
        'walls': [(1,1),(0,2)],
        'start': (0,0),
        'goal': (1,2),
        'theta': 0
        } 

env = gym.make('TurtleRobotEnv-v1', **setup)

epsilon=1.0

def choose_action(epsilon,state):
    if np.random.random() <= epsilon:
        return random.randint(0,2)
    else:
        return np.random.choice((np.argwhere(Q[state,:] == np.amax(Q[state,:]))).flatten())

Q = np.zeros([288, 3])
lr = 0.15
y = 0.99
eps = 500
visited_states=[]
list_acciones=[]
print(env.reset())
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

f=open('AccionesQLeaning.txt','w')
f.write(str(list_acciones))
f.close()

x=[i for i in range(len(list_acciones))]

plt.plot(x,list_acciones)
plt.grid()
plt.show()

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
