from gym.envs.registration import register

register(
    id='TurtleRobotEnv-v0',
    entry_point='turtle_robot_gym.envs:TurtleRobotEnv_v0',
)

register(
    id='TurtleRobotEnv-v1',
    entry_point='turtle_robot_gym.envs:TurtleRobotEnv_v1',
)
register(
    id='TurtleRobotEnv-v1_2',
    entry_point='turtle_robot_gym.envs:TurtleRobotEnv_v1_2',
)
register(
    id='TurtleRobotEnv-v1_3',
    entry_point='turtle_robot_gym.envs:TurtleRobotEnv_v1_3',
)
register(
    id='TurtleRobotEnv-v1_4',
    entry_point='turtle_robot_gym.envs:TurtleRobotEnv_v1_4',
)
