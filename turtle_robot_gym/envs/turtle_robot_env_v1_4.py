import gym
import numpy as np
from gym.spaces import Discrete,MultiDiscrete
import sys
sys.path.append('../..')
from client_dqn import tcp_dqn
class TurtleRobotEnv_v1_4(gym.Env):
	
	def __init__(self, **kwargs):
		super().__init__()
		# dimensions of the grid and walls
		self.width = kwargs.get('width')
		self.height = kwargs.get('height')  
		self.walls = kwargs.get('walls')	
		self.start = kwargs.get('start') 
		self.goal = kwargs.get('goal')
		self.init_theta = kwargs.get('theta')
		self.numero_episodios=0
	  
		# define the maximum x and y values
		self.max_x = self.width - 1
		self.max_y = self.height - 1

		# there are 4 possible actions: move,rotate left,rotate right or stay in same state
		self.action_space = Discrete(4)		  

		# the observation will be the coordinates of Baby Robot			
		self.observation_space = MultiDiscrete([self.width, self.height, 4])
										
		# Turtle Robot's position in the grid
		self.x = self.start[0]
		self.y = self.start[1] 
		self.theta = self.init_theta
		
		# Sensor's readings
		self.get_sensor_readings()
			
	def get_sensor_readings(self):	
		self.right = (self.theta == 0 and self.y + 1 <= self.max_y and (self.x,self.y+1) not in self.walls) or (self.theta == 1 and self.x + 1 <= self.max_x and (self.x+1,self.y) not in self.walls) or (self.theta == 2 and self.y - 1 >= 0 and (self.x,self.y-1) not in self.walls) or (self.theta == 3 and self.x - 1 >= 0 and (self.x-1,self.y) not in self.walls)
		self.right=1 if self.right==True else 0
		self.front = (self.theta == 0 and self.x - 1 >= 0 and (self.x-1,self.y) not in self.walls) or (self.theta == 1 and self.y + 1 <= self.max_y and (self.x,self.y+1) not in self.walls) or (self.theta == 2 and self.x + 1 <= self.max_x and (self.x+1,self.y) not in self.walls) or (self.theta == 3 and self.y - 1 >= 0 and (self.x,self.y-1) not in self.walls) 
		self.front=1 if self.front==True else 0
		self.left = (self.theta == 0 and self.y - 1 >= 0 and (self.x,self.y-1) not in self.walls) or (self.theta == 1 and self.x - 1 >= 0 and (self.x-1,self.y) not in self.walls) or (self.theta == 2 and self.y + 1 <= self.max_y and (self.x,self.y+1) not in self.walls) or (self.theta == 3 and self.x + 1 <= self.max_x and (self.x+1,self.y) not in self.walls)
		self.left=1 if self.left==True else 0

		self.rel_goal = ''
		if self.x < self.goal[0]:
			# North
			self.rel_goal = 'South'
			self.rel_goal=10
		elif self.x > self.goal[0]:
			# South
			self.rel_goal = 'North'
			self.rel_goal=20
		elif self.x == self.goal[0]:
			# Equal
			self.rel_goal = 'Equal'
			self.rel_goal=30
		if self.y < self.goal[1]:
			# East
			#self.rel_goal = self.rel_goal + '-East'
			self.rel_goal+=1
		elif self.y > self.goal[1]:
			# West
			#self.rel_goal = self.rel_goal + '-West'
			self.rel_goal+=2
		elif self.y == self.goal[1]:
			#self.rel_goal = self.rel_goal + '-Equal'
			self.rel_goal+=3
			
		self.distance = abs(self.x - self.goal[0]) + abs(self.y - self.goal[1])
		
		return np.array([self.right, self.front, self.left, self.rel_goal, self.distance])

	def take_action(self, action):
		self.old_state = self.get_sensor_readings()
		if action == 0: # Move
			if self.theta == 0: # Up
				if self.x - 1 >= 0 and (self.x-1,self.y) not in self.walls:
					self.x = self.x - 1
			elif self.theta == 1 and (self.x,self.y+1) not in self.walls: # Right
				if self.y + 1 <= self.max_y:
					self.y = self.y + 1
			elif self.theta == 2 and (self.x+1,self.y) not in self.walls: # Down
				if self.x + 1 <= self.max_x:
					self.x = self.x + 1
			elif self.theta == 3: # Left
				if self.y - 1 >= 0 and (self.x,self.y-1) not in self.walls:
					self.y = self.y - 1
		elif action == 1: # Rotate left
			if self.theta == 0:
				self.theta = 3
			else:
				self.theta = self.theta - 1
		elif action == 2: # Rotate right
			if self.theta == 3:
				self.theta = 0
			else:
				self.theta = self.theta + 1
		
		return self.get_sensor_readings()

	def step(self, action):		
		if self.numero_episodios==1500:
			a=tcp_dqn(['step',int(action),[self.x,self.y],self.theta])
			print('AAAAAAAAA',a)
			obs=a[0]
			self.x=a[1][0]
			self.y=a[1][1]
			self.theta=a[2]
			x=tcp_dqn(['reward',[self.x,self.y]])
			reward=int(x[0])
			done=x[1]
		else:
			obs = self.take_action(action)
			reward = -1
			# set the 'done' flag if we've reached the goal
			done = (self.x == self.goal[0]) and (self.y == self.goal[1])
			if done:
				reward = 10
		info = {'target_reached':done}
		
		return obs, reward, done, info 

	def reset(self):
		# reset Turtle Robot's position in the grid
		self.x = self.start[0]
		self.y = self.start[1]
		self.theta = self.init_theta
		self.numero_episodios+=1
					
		return self.get_sensor_readings()	  
  
	def render(self, action=0, reward=0):
		print("robot start: ", self.old_state[0], self.old_state[1], self.old_state[2], self.old_state[3], self.distance, " action: ", action, " robot end: ", self.right, self.front, self.left, self.rel_goal, self.distance, " reward: ", reward)

