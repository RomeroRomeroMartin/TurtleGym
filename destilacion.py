import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
from keras.layers import Input, Flatten, Dense
import ast
# Load saved DQN model weights
#model = keras.models.load_model('dqn_weights.h5')

# Define the distilled model architecture
#tf.config.run_function_eagerly(True)
def Kulback_Leibler_loss(y_true,y_pred):
    tau=1
    e_x = tf.exp(y_true/tau - tf.reduce_max(y_true/tau))
    softmax1=e_x / tf.reduce_sum(e_x,keepdims=True)

    e_x = tf.exp(y_pred - tf.reduce_max(y_pred))
    softmax2=e_x / tf.reduce_sum(e_x,keepdims=True)

    epsilon = 1e-17  # valor para reemplazar los valores de softmax2 menores a este límite
    softmax2 = tf.where(softmax2 < epsilon, epsilon, softmax2)  # reemplaza los valores menores a epsilon

    return tf.reduce_sum(softmax1*tf.math.log(softmax1/softmax2))

#Laberintos 3x3 y 4x4
model = keras.Sequential()
#Input is 1 observation vector, and the number of observations in that vector 
model.add(Input(shape=(1,5)))  
model.add(Flatten())
#Hidden layers with 24 nodes each
model.add(Dense(12, activation='relu'))                             
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='linear'))

#Laberintos 5x5 y 6x6
'''model = keras.Sequential()
#Input is 1 observation vector, and the number of observations in that vector 
model.add(Input(shape=(1,5)))  
model.add(Flatten())
#Hidden layers with 24 nodes each
model.add(Dense(24, activation='relu'))                             
#model.add(Dense(92, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(4, activation='linear'))''' 

# Compile the distilled model with KL divergence loss
model.compile(optimizer='adam', loss=Kulback_Leibler_loss)

# Load the dataset of states and action probabilities
estados=[]
f1=open('data/InputsComite3x3.txt','r')
lineas=f1.readlines()
for linea in lineas:
    l=np.array([])
    lista_linea=linea.split(' ')
    for elemento in lista_linea:
        try:
            l=np.append(l,int(elemento))
            #print(l)
        except:
            pass
    estados.append(np.reshape(l,(1,5)))
    #estados.append(l)
estados=np.array(estados)
#print(estados)
f1.close()


f2=open('data/OutputsComite3x3.txt','r')
predicciones=[]
lineas=f2.readlines()
for linea in lineas:
    l=np.array([])
    lista_linea=linea.split(' ')
    for elemento in lista_linea:
        try:
            l=np.append(l,float(elemento))
        except:
            pass
    predicciones.append(l)
    #predicciones=np.concatenate(predicciones,l)
predicciones=np.array(predicciones)
#print(predicciones)
f2.close()

# Train the distilled model on the dataset
model.fit(estados, predicciones,verbose=1,epochs=500)

# Save the distilled model weights
model.save('models/3x3comite_weights_12x2.h5')