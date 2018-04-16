from setuptools import setup
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gazeboschool'))

setup(name='gazeboschool-gym',
      version='0.0.1',
      install_requires=[
		'gym',
		'numpy',
		'requests',
		'six',
		'pyglet',
		'scipy',
		'matplotlib',
		'defusedxml',
		'scikit-image',
      ],
      description='The OpenAI Gym for robotics: A toolkit for developing and comparing your reinforcement learning agents using Gazebo and ROS.',
      author='Erle Robotics',
      package_data={'gazeboschool-gym': ['envs/assets/launch/*.launch', 'envs/assets/worlds/*']},
)
