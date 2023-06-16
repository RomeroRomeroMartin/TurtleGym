## INSTRUCCIONES PARA LA EJECUCION DEL PAQUETE ROS

Este directorio es un paquete de ros necesario para ejecutar el entrenamiento y movimientos en el Turtlebot real. Este paquete se llama `turtle_tfg` y dentro contiene varias carpetas. La primera es `data`, que contiene dentro el archivo `AccionesQlearning.txt`, que nos permitirá generar la gráfica de evolución del entrenamiento de Q-Learning que se ha llevado a cabo en el propio Turtlebot. Para generar esta gráfica, se puede utilizar el programa Python mencionado anteriormente `plotea.py`. 

La siguiente carpeta dentro del paquete es `models`, en la que se encuentra el modelo de Q-Learning tras finalizar todos los entrenamientos `Qlear3x3.pkl`, el modelo tras 1 episodio de entrenamiento `Qlear3x3_1.pkl` y tras 20 episodios `Qlear3x3_20.pkl`. 
Por último, en la carpeta “src” se encentran los programas necesarios para el entrenamiento y movimiento del robot. Los programas son los siguientes: 

    - movimiento.py: es el programa que entrena el algoritmos de Q-Learning en el Turtlebot. 

    - test.py: es el programa para testear el modelo de Q-Learning entrenado. 

    - train_dqn.py: este programa sirve para entrenar la DQN en el Turtlebot real. Además de este programa es necesario lanzar el programa `dqn_turtle.py` de la carpeta `1_CP_TurtleGym`. 

    - dqn.py: este programa sirve para testear el modelo de DQN entrenado. Para lanzar este programa es necesario también lanzar el programa `servidor.py`, que creará un servidor TCP con el Turtlebot para la comunicación entre ambos. 

    - turtlebot_communication.py: es un programa auxiliar para crear la comunicación entre Turtlebot y ordenador. 

 

Para lanzar cualquier programa que se encuentre dentro de la carpeta `src`, se puede lanzar de la siguiente forma: 

		rosrun nombre_programa.py 

Hay que tener en cuenta que antes de lanzar cualquier programa en el Turtlebot real, primero es necesario tener lanzado el nodo que nos permitirá acceder a la velocidad de los motores y el nodo que nos permitirá acceder a la información del lidar. Para lanzar estos nodos es necesario ejecutar en una terminal los siguientes comandos (cada uno en una terminal): 

		roslaunch kobuki_node minimal.launch 

		roslaunch rplidar_ros rplidar_a3.launch 
