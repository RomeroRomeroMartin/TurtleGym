#EXPLICACION DE LOS DOCUMENTOS DEL DIRECTORIO:

En este directorio se guardan los modelos obtenidos de los entrenamientos. Dentro de esta carpeta están los modelos de Q-Learning nombrados de la siguiente forma: `QlearXXX.pkl`, donde XXX es el laberinto en el que se ha entrenado. Por otro lado, están los modelos de DQN entrenados individualmente, la destilación de estos modelos y la destilación de los ensembles. Primeramente, se indica el laberinto en el que está entrenado, seguido de esto, se indica si son modelos individuales (`turtle_weights`), modelos destilados (`distilled_weights`) o modelos destilados del ensemble (`comité_weights`). Por último, se indica el tamaño de la red usada, y en algunos casos de la destilación, el tamaño de la red original y el tamaño de la red destilada. 

Para usar estos modelos existen varios programas que se encuentran en el directorio principal, y que están explicados en el archivo README.md correspondiente
