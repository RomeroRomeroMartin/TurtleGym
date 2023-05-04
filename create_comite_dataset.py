import gym
import turtle_robot_gym
import numpy as np
import random
from keras import Sequential
from keras.layers import Input, Flatten, Dense
import rl
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
import tensorflow as tf
import pickle


class create_dqn():
    def __init__(self,architecture,weights):
        self.architecture=architecture
        self.weights=weights


    def create_model(self):
        #Feed-Forward Neural Network Model for Deep Q Learning (DQN)
        model = Sequential()
        #Input is 1 observation vector, and the number of observations in that vector 
        model.add(Input(shape=(1,5)))  
        model.add(Flatten())
        for i in range(len(self.architecture)):
            #Hidden layers
            model.add(Dense(self.architecture[i], activation='relu'))                                
        model.add(Dense(4, activation='linear'))

        return model
    
    def create_agent(self):
        model=self.create_model()

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

        dqn.load_weights(self.weights)
        return dqn

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


def get_prediction(state,list_dqn,list_qlear):
    state_qlear= list(map(str, state))
    state_qlear=''.join(state_qlear)
    final_prediction=[0,0,0]
    for i in range(len(list_dqn)):
        dqn_prediction=list_dqn[i].compute_q_values(np.reshape(state,(1,5)))
        normalized_dqn_predictions = [(x - min(dqn_prediction)) / (max(dqn_prediction) - min(dqn_prediction)) for x in dqn_prediction]
        for i in range(len(final_prediction)):
            final_prediction[i]+=normalized_dqn_predictions[i]
        
    for i in range(len(list_qlear)):
        Q=list_qlear[i][0]
        visited_states=list_qlear[i][1]
        indice=visited_states.index(state_qlear)
        qlear_prediction=Q[indice]
        normalized_qlear_predictions=[(x - min(qlear_prediction)) / (max(qlear_prediction) - min(qlear_prediction)) for x in qlear_prediction]
        for i in range(len(final_prediction)-1):
            final_prediction[i]+=normalized_qlear_predictions[i]
    
    return final_prediction



list_qlear=[]
list_dqn=[]

#CREAMOS DQN
modeldqn=create_dqn([24,24],'models/3x3_turtle_weights.h5').create_agent()
list_dqn.append(modeldqn)
modeldqn=create_dqn([12,12],'models/3x3distilled_weights.h5').create_agent()
list_dqn.append(modeldqn)

#CREAMOS Q-LEARNING
with open('data/Qlear3x3.pkl', 'rb') as f:
    qlear = pickle.load(f)

list_qlear.append(qlear)


f1=open('data/estados.txt','w')
f2=open('data/predicciones.txt','w')
contador=0
OldState=env.reset()
done=False
while not done:

    predictions= get_prediction(OldState,list_dqn,list_qlear)
    f1.write(str(OldState[0])+' '+str(OldState[1])+' '+str(OldState[2])+' '+str(OldState[3])+' '+str(OldState[4])+' '+'\n')
    f2.write(str(predictions[0])+' '+str(predictions[1])+' '+str(predictions[2])+' '+str(predictions[3])+'\n')
    action=np.argmax(predictions)
    new_state, reward, done, info = env.step(action)
    OldState=new_state
    env.render(action=action, reward=reward)

print(contador)
f1.close()
f2.close()






