import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


f = open('data/500SimulacionesQlear6x6.txt','r')
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
plt.title('Media y desviación tras 500 entrenamientos de 500 episodios en tablero 6x6')
plt.xlabel('Número de episodio')
plt.ylabel('Número de acciones por episodio')
plt.legend()
plt.grid()
plt.show()
