import numpy as np
import matplotlib.pyplot as plt
import random

f = open('epsilon.txt')
lineas = f.read()

datos=lineas.split(',')

ent=[]
x=[]
for i in range(len(datos)):
    ent.append(float(datos[i]))
    x.append(i)


plt.plot(x,ent)
plt.grid()
plt.show()


f = open('ticks.txt')
lineas = f.read()

datos=lineas.split(',')

ent=[]
x=[]
for i in range(len(datos)):
    ent.append(float(datos[i]))
    x.append(i)


plt.plot(x,ent)
plt.grid()
plt.show()
