import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


f = open('../turtle_tfg/data/AccionesQlearning.txt','r')
lineas = f.readlines()
datos=[]
datos=[eval(lineas[i]) for i in range(len(lineas))]

medias=[]
std=[]
for j in range(len(max(datos,key=len))):
    aux=[]
    #print(len(datos[j]))
    for k in range(len(datos)):
        #print(len(datos[j]))
        try:
            aux.append(datos[k][j])
        except:
            pass
    medias.append(np.mean(aux))
    std.append(np.std(aux))

x=[i for i in range(len(medias))]
plt.plot(x,medias,label='Means')
plt.fill_between(range(len(medias)), np.subtract(medias, std), np.add(medias, std),alpha=0.2, edgecolor='blue', facecolor='blue',label='Standard Desviation')
plt.title('Media y desviación tras 500 entrenamientos de DQN en tablero 3x3 con turtlebot',fontsize=30)
plt.xlabel('Número de episodio',fontsize=25)
plt.ylabel('Número de acciones por episodio',fontsize=25)
plt.legend()
plt.grid()
plt.show()
