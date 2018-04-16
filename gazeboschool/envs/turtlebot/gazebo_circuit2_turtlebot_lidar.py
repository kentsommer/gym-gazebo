import gym
import rospy
import roslaunch
import time
import numpy as np
import os

from gym import utils, spaces
from gazeboschool.envs import gazebo_env
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import *
from gazebo_msgs.msg import *

from tf import TransformListener

from gym.utils import seeding

class GazeboCircuit2TurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.delete_proxy = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        self.spawn_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.prev_distance = None
        self.prev_action = [0., 0.]
        self.time_step = 0
        self.max_time_steps = 250
        self.scan_dim = 20
        self.obs_dim = self.scan_dim + 4
        self.min_range = 0.2

        # Load target xml for respawning
        target_xml_path = os.path.expandvars('$GAZEBO_MODEL_PATH/Target/model.sdf')
        with open(target_xml_path, 'r') as f:
            self.target_xml = f.read()

        low = np.array([0.0, -1.0])
        high = np.array([1.0, 1.0])
        self.action_space = spaces.Box(low, high)

        high = np.ones(self.obs_dim)
        low = np.zeros(self.obs_dim)
        self.observation_space = spaces.Box(low, high)
        self.reward_range = (-np.inf, np.inf)

        self._seed()


    # def discretize_observation(self,data):
    #     discretized_ranges = []
    #     min_range = 0.2
    #     done = False
    #     mod = len(data.ranges) / 20 
    #     for i, item in enumerate(data.ranges):
    #         if (i%mod==0):
    #             if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
    #                 discretized_ranges.append(20.0)
    #             elif np.isnan(data.ranges[i]):
    #                 discretized_ranges.append(0.0)
    #             else:
    #                 discretized_ranges.append(data.ranges[i])
    #         if (min_range > data.ranges[i] > 0):
    #             done = True
    #     x = np.asarray(discretized_ranges)
    #     new_ranges = (x-0.0) / (20.0-0.0)
    #     return new_ranges,done


    def discretize_observation(self,data):
        done = False
        oldmin, oldmax = 0.06, 15.
        oldrange = oldmax - oldmin
        newmin, newmax = 0., 1.
        newrange = newmax - newmin
        scale = newrange / oldrange

        for i in range(len(data.ranges)):
            # Check if we are too close to obstacle
            if (self.min_range > data.ranges[i] > 0):
                done = True

        normalized_obs = [(v - oldmin) * scale + newmin for v in data.ranges]
        obs = np.asarray(normalized_obs)
        return obs, done


    def _seed(self, seed=None):
        print("seed is: {0}".format(seed))
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _get_model_pose(self, model_name):
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model = GetModelStateRequest()
        model.model_name = model_name
        position = get_model_state(model).pose.position
        result = np.asarray([position.x, position.y])
        return result


    def _get_data(self):
        scan = None
        positions = None
        finished = scan != None # and positions != None
        while not finished:
            try:
                robot_frame = 'mobile_base'
                target_frame = 'Target'
                scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                robot_pose = self._get_model_pose(robot_frame)
                target_pose = self._get_model_pose(target_frame)
                finished = scan != None # and positions != None
            except:
                pass

        return scan, robot_pose, target_pose


    def _get_valid_target_state(self):
        x = np.random.uniform(-8.4, 5.6)
        y = np.random.randint(-8.304, 8.49)

        target_state = ModelState()
        target_state.model_name = 'Target'
        target_state.pose.position.x = x
        target_state.pose.position.y = y
        target_state.pose.position.z = 0
        target_state.pose.orientation.x = 0
        target_state.pose.orientation.y = 0
        target_state.pose.orientation.z = 0
        target_state.pose.orientation.w = 0
        target_state.twist.linear.x = 0.0
        target_state.twist.linear.y = 0
        target_state.twist.linear.z = 0
        target_state.twist.angular.x = 0.0
        target_state.twist.angular.y = 0
        target_state.twist.angular.z = 0.0
        target_state.reference_frame = 'world'
        return target_state


    def _spawn_target(self):
        x = np.random.randint(1, 13)
        y = np.random.randint(0, 11)

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = 0
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 0

        req = SpawnModelRequest()
        req.model_name = "Target"
        req.model_xml = self.target_xml
        req.initial_pose = pose

        res = self.spawn_proxy(req)


    def _build_state(self, scan, robot_pose, target_pose):
        relative_pose = self._get_relative_pose(robot_pose, target_pose)
        state =  np.concatenate((scan, relative_pose, self.prev_action), axis=0)
        return state


    def _get_relative_pose(self, robot_pose, target_pose):
        return target_pose - robot_pose


    def _get_distance(self, robot_pose, target_pose):
        distance = np.linalg.norm(robot_pose - target_pose)
        return distance


    def _update_distance(self, robot_pose, target_pose):
        distance = self._get_distance(robot_pose, target_pose)
        self.prev_distance = distance


    def _reward(self, robot_pose, target_pose, action, done):
        distance = self._get_distance(robot_pose, target_pose)

        beta = 5 # 5 is OKAY not GREAT

        if done or self.time_step > self.max_time_steps:
            done = True
            return -35, done # -25 is OKAY not GREAT
        elif distance < 1.5:
            done = True
            return 100, done # 100 is OKAY not GREAT

        reward = beta * np.e * (self.prev_distance - distance)

        return reward, done


    def _step(self, action):
        self.time_step += 1
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)

        scan, robot_pose, target_pose = self._get_data()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        scan, done = self.discretize_observation(scan)
        state = self._build_state(scan, robot_pose, target_pose)

        reward, done = self._reward(robot_pose, target_pose, action, done)

        self._update_distance(robot_pose, target_pose)
        self.prev_action = action
        return state, reward, done, {}

    def _reset(self):
        self.time_step = 0
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")




        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            target_state = self._get_valid_target_state()
            set_model_state(target_state)
        except(rospy.ServiceException) as e:
            print ("/gazebo/set_model_state service call failed")

        # rospy.wait_for_service('/gazebo/delete_model')
        # try:
        #     res = self.delete_proxy("Target")
        # except (rospy.ServiceException) as e:
        #     print('/gazebo/delete_model service call failed')

        # rospy.wait_for_service('/gazebo/spawn_urdf_model')
        # try:
        #     self._spawn_target()
        # except (rospy.ServiceException) as e:
        #     print('/gazebo/spawn_sdf_model service call failed')



        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        scan, robot_pose, target_pose = self._get_data()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        scan, done = self.discretize_observation(scan)
        state = self._build_state(scan, robot_pose, target_pose)

        # Set initial previous distance
        self._update_distance(robot_pose, target_pose)
        self.prev_action = [0., 0.]

        return state