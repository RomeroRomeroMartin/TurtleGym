# PASOS A SEGUIR PARA LA EJECUCIÓN DEL CÓDIGO

A continuación se describen los pasos necesarios para poder ejecutar el código proporcionado, a modo de "manual de usuario", para todas las personas que deseen replicar todo lo que se ha hehco en este trabajo o incluso continuar con la línea de investigación.

Todos los programas son programas python y se ejecutan de la siguiente manera:

		python3 nombre_programa.py

## Programas principales:
A continuación se explicarán las funcionalidades de cada programa:

	- plotea.py: sirve para extraer las gráficas de evolución de los algoritmos Q-Learning y DQN. Los archivos .txt con la información necesaria para extraer las gráficas se encuentran en el directorio `data`. En el archivo README.md del directorio data se explica el contenido que hay dentro.
	- testQlear.py: este programa sirve para testear el modelo de Q-Learning aprendido. El modelo se encuentra en el directorio `models`. En el arhivo README.md del directorio se explica el contenido que hay dentro.
	- test.py: este programa sirve para testar el modelo de DQN individual. El modelo se encuentra en el directorio `models`. En el arhivo README.md del directorio se explica el contenido que hay dentro.
	- test_destilacion.py: este programa sirve para testar los modelos de DQN destilados. Los modelos se encuentran en el directorio `models`. En el arhivo README.md del directorio se explica el contenido que hay dentro.
	- main.py: con este programa se entrenan los modelos de Q-Learning. Los modelos se guardan en el directorio `models`.
	- mainDQN.py: con este programa se entrenan los modelos de DQN. Los modelos se guardan en el directorio `models`.
	- destilacion.py: con este programa se obtienen los modelos destilados. Para entrenar cada modelo se necesitan los inputs y outputs que se encuentran en `data` y el modelo final se guarda en `models`. 
	- create_comite_weights.py: con este programa se crean los archivos (que se guardan en `data`) necesarios para poder destilar los ensembles. 
	- client_dqn.py: es un programa auxiliar que utiliza el programa `turtle_robot_env_v1_4.py` para crear una conexión TCP con el Turtlebot. 
	- dqn_turtle.py y servidor.py: son programas que se utilizaron para el entrenamiento en el Turtlebot real.


## Programas para crear el entorno de simulación gym

En el directorio `turtle_robot_gym` se encuentran los programas necesarios para la creación del entorno de simulación gym. Concretamente dentro del directorio `envs` existen varias versiones del entorno. Principalmente se han utilizado dos:

	- turtle_robot_env_v1_2.py: es el entorno de simulación mayormente utilizado para obtener todos los resultados que no implican el uso del Turtlebot real.
	- turtle_robot_env_v1_4.py: es el entorno utilizado para entrenar la DQN en el Turtlebot real. Es prácticamente igual al anterior, pero en el se implementa lo necesario para la comunicación con el Turtlebot.
	
