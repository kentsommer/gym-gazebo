import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# Turtlebot envs
register(
    id='TurtlebotNavLidar-v0',
    entry_point='gazeboschool.envs.turtlebot:GazeboCircuit2TurtlebotLidarEnv',
    max_episode_steps=10000,
    # More arguments here
)

register(
    id='TurtlebotNavDepth-v0',
    entry_point='gazeboschool.envs.turtlebot:GazeboCircuit2TurtlebotDepthEnv',
    max_episode_steps=10000,
    # More arguments here
)