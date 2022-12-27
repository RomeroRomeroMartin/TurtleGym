from gym.envs.registration import register

register(
    id='TurtleRobotEnv-v0',
    entry_point='turtle_robot_gym.envs:TurtleRobotEnv_v0',
)

register(
    id='TurtleRobotEnv-v1',
    entry_point='turtle_robot_gym.envs:TurtleRobotEnv_v1',
)
