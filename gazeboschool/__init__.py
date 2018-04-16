import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# Turtlebot envs
register(
    id='GazeboCircuit2TurtlebotLidar-v0',
    entry_point='gazeboschool.envs.turtlebot:GazeboCircuit2TurtlebotLidarEnv',
    max_episode_steps=10000,
    # More arguments here
)