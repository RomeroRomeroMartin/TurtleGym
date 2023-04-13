import gym
import numpy as np
from gym.spaces import Discrete,MultiDiscrete

class TurtleRobotEnv_v0(gym.Env):
	
	def __init__(self, **kwargs):
		super().__init__()
		# dimensions of the grid and walls
		self.width = kwargs.get('width')
		self.height = kwargs.get('height')  
		self.walls = kwargs.get('walls')	
		self.start = kwargs.get('start') 
		self.goal = kwargs.get('goal')
		self.init_theta = kwargs.get('theta')
	  
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

	def take_action(self, action):
		self.old_pos = [self.x, self.y, self.theta]
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
		
		return np.array([self.x, self.y, self.theta])

	def step(self, action):		
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
		return np.array([self.x,self.y,self.theta])	  
  
	def render(self, action=0, reward=0):
		print("robot start: ", self.old_pos[0], self.old_pos[1], self.old_pos[2], " action: ", action, " robot end: ", self.x, self.y, self.theta, " reward: ", reward)
