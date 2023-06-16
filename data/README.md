# EXPLICACIÓN DE LOS DOCUMENTOS DEL DIRECTORIO

## Archivos `.txt` para obtener las gráficas de evolución:
Los archivos `.txt` para obtener las gráficas de evolución están nombrados de la siguiente manera: `XXXSimulacionesYYYZZZ.txt` o `XXXSimulacionesYYYZZZ_AAA.txt`, donde: 

    - XXX son el número de simulaciones (30 o 500). 

    - YYY es el algoritmo utilizado (“Qlear” para Q-Learning, y “DQN” para DQN). 

    - ZZZ indica el laberinto sobre el que se entrenado el algoritmo (3x3, 4x4, 5x5 o 6x6). 

    - AAA indica el tamaño de la red, de la forma CxBB, donde C es el número de capas ocultas y BB es el número de neuronas por capa. 

Estas gráficas pueden obtenerse ejecutando el programa Python `plotea.py`, modificando los archivos de datos correspondientes en el programa. 

## Archivos `.txt` para la destilación de las redes (tanto individuales como los ensembles):

Para cada modelo destilado existen dos archivos, uno con las entradas y otro con las salidas. Estos archivos tienen una forma del tipo “XXXYYYZZZ.txt” o “XXXYYYZZZ_AAA.txt” donde: 

    - XXX indica si es un archivo de entrada con el nombre “Inputs” o un archivo de salida con el nombre “Outputs”.

    - YYY indica si es una destilación de un modelo individual con “Destilación” o si es una destilación de un ensemble con “Comité”. 

    - ZZZ indica el laberinto en el que se ha hecho el entrenamiento algoritmo (3x3, 4x4, 5x5 o 6x6). 

    - AAA indica el tamaño de la red que se destila. 

En caso de haber un archivo don un nombre que no siga este patrón, su función quedará explicada en el propio nombre del archivo. 


