import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

f = open('AccionesQLeaning.txt','r')
lineas = f.read()

datos=eval(lineas)

x=[i for i in range(len(datos))]

plt.plot(x,datos)
plt.grid()
plt.show()



f = open('AccionesDQN.txt','r')
lineas = f.read()

datos=eval(lineas)
x=[i for i in range(len(datos))]

plt.plot(x,datos)
plt.grid()
plt.show()
