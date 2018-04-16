import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2

from gym import utils, spaces
from gazeboschool.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

from gym.utils import seeding

class GazeboCircuit2TurtlebotDepthEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.bridge = CvBridge()
        self.obs_dim = 20

        low = np.array([0.05, -0.3])
        high = np.array([0.5, 0.3])
        self.action_space = spaces.Box(low, high)
        # self.action_space = spaces.Discrete(3) #F,L,R

        high = np.ones(self.obs_dim)
        low = np.zeros(self.obs_dim)
        self.observation_space = spaces.Box(low, high)
        self.reward_range = (-np.inf, np.inf)


        self._seed()

    def get_observation(self, data_lidar, data_depth):
        min_range = 0.2
        done = False
        for i, item in enumerate(data_lidar.ranges):
            if (min_range > data_lidar.ranges[i] > 0):
                done = True

        cv_image = self.bridge.imgmsg_to_cv2(data_depth, "32FC1")
        cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        cv_final = cv_image_norm
        # cv_final = cv2.resize(cv_image_norm, (128, 128))
        cv2.imshow("Depth Image", cv_final)
        cv2.waitKey(1)
        return cv_final, done

    def _seed(self, seed=None):
        print("seed is: {0}".format(seed))
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)

        data = None
        image_data = None
        while data is None and image_data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                image_data = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.get_observation(data, image_data)

        if not done:
            if action[0] > 0.15 and abs(action[1]) <= 0.025:
                reward = 10
            elif action[0] > 0.1:
                reward = 1
            else:
                reward = -0.5

            # if np.any(np.less_equal(state, 0.1)):
            #     reward -= 0.5
            # if action == 0:
            #     reward = 5
            # else:
            #     reward = 1
        else:
            reward = -200

        return state, reward, done, {}

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        data = None
        image_data = None
        while data is None and image_data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                image_data = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.get_observation(data, image_data)

        return state