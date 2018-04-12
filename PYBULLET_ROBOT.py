'''
This class is written as interface to the pybullet physical environment
'''

import sys
import numpy as np
import pybullet as p
from IMAGE_REWARDS import *
import time
import os

PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

# get the current directory
path = os.getcwd()
robot_path = path+'/kuka_lwr/kuka.urdf'
ball_path = path+'/ball/sphere_small.urdf'
ground_path = path + "/floor/plane100.urdf"

# pybullet parameters setup
robot_orn_init = [0,0,0]   # siwei should hardcode this
robot_pos_init = [0,0,0]
robot_joint_init = [0, -0.488, 0, 0.307, 0, -0.8, 0]
ball_pos_init = [1.5,0,0.9]
ground_pos = [0,0,0]
j_d = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]  # joint damping

# camera parameter setup
XSIZE = 128
YSIZE = 128


class PybulletRobot:

    def __init__(self, mode='DIRECT', ff=None):

        # state space and action space
        self.state_space = np.zeros(17) # x, y, radius, velocityx7, toruqex7
        self.action_space = np.zeros(7)  # torquex7

        # camera frame rate
        self.frame_rate = 20

        # inistialize pybullet physics environment
        if mode is 'GUI':
            p.connect(p.GUI)
            p.setRealTimeSimulation(1)
        else:
            p.connect(p.DIRECT)
        self.mode = mode
        self.ff = ff
        p.resetSimulation()

        # action low and high mapped to 1
        self.action_low = 200
        self.action_high = 500

        # load the two items
        self.robot_id = p.loadURDF(robot_path, robot_pos_init, p.getQuaternionFromEuler(robot_orn_init), useFixedBase=True)
        self.ball_id = p.loadURDF(ball_path, ball_pos_init)
        self.ground_id = p.loadURDF(ground_path, ground_pos, useFixedBase=True)

        # get the joint number and the initial joint state
        self.robot_joint_num = p.getNumJoints(self.robot_id)
        self.init_radius = 0

    def _state_space_dim(self):
        return self.state_space.shape[0]

    def _action_space_dim(self):
        return self.action_space.shape[0]

    # update the state spacex17
    def _update_state(self, reset=False):
        image_frame = self._take_picture()
        center, radius = compute_center_and_size(image_frame)
        if self.ff is not None:
            self.ff.write('The center is %s, the radius is %i \n' %(center, radius))
        if center is None:
            center = [-1, -1]
        res = p.getJointStates(self.robot_id, np.arange(self.robot_joint_num))
        self.state_space[0:2] = list(center)  # update center - x, y
        self.state_space[2] = radius          # update radius
        # update velocity and torque
        for i in range(self.robot_joint_num):
            self.state_space[3+i] = res[i][1]
            self.state_space[10+i] = res[i][3]
        self.state_space[10] = 0
        self.state_space[16] = 0
        if reset:
            self.init_radius = radius
        return center

    # maps (-1, 0) to (-500, -200), (0, 1) to (200, 500)
    def _map_action(self, action):
        c_action = np.clip(action, -1, 1)
        return self.action_low * np.sign(c_action) + (self.action_high - self.action_low) * np.array(c_action)

    # reset the robot to the initial state
    def _reset(self):
        # restore the state space to original ones
        self.state_space = np.zeros(17)
        # torque
        self.action_space = np.zeros(7)
        #p.setRealTimeSimulation(0)
        p.setGravity(0,0,0)
        # reset the robot's joint states, test resetJointStates
        for i in range(self.robot_joint_num):
            for j in range(20):
                p.resetJointState(self.robot_id, i, robot_joint_init[i])
                if self.mode is 'DIRECT':
                    p.stepSimulation()
        print('moving back to the original position')
        for j in range(5):
                p.resetBasePositionAndOrientation(self.ball_id, ball_pos_init, p.getQuaternionFromEuler(robot_orn_init))
                if self.mode is 'DIRECT':
                    p.stepSimulation()
        print('ball back to the original position')
        # take the picture before ball moves
        self._update_state(True)  # true = update the init_radius
        p.setGravity(0,0,-9.8)
        time.sleep(0.5)
        return self.state_space

    # check if ball or robot collides with the ground
    def _check_collision(self):
        if len(p.getContactPoints(self.ball_id, self.ground_id)) > 0: #or len(p.getContactPoints(self.robot_id, self.ground_id)) > 0:
            return True
        else:
            return False

    # step by given action(torque)
    def _step(self, action):
        mapped_action = self._map_action(action)
        for i in range(3):
            for j in range(1, self.robot_joint_num-1):
                p.setJointMotorControl2(self.robot_id,j,p.TORQUE_CONTROL,force=mapped_action[j+1])
            if self.mode is 'DIRECT':
                p.stepSimulation()
            time.sleep(0.001)  # naive computation
        # update the state
        center = self._update_state()
        # compute the reward
        r = compute_reward(self.state_space[0:2], self.state_space[2], XSIZE, YSIZE, self.init_radius)
        return (self.state_space, r, center) # s_ and reward


    # perform taking pictures
    def _take_picture(self):
        # get the last link's position and orientation
        Pos, Orn = p.getLinkState(self.robot_id, self.robot_joint_num-1)[:2]
        # Pos is the position of end effect, orn is the orientation of the end effect
        rotmatrix = p.getMatrixFromQuaternion(Orn)
        # distance from camera to focus
        distance = 0.2
        # where does camera aim at
        camview = list()
        for i in range(3):
            camview.append(np.dot(rotmatrix[i*3:i*3+3], (0, 0, distance)))
        tagPos = np.add(camview,Pos)
        #p.removeBody(kukaId)
        viewMatrix = p.computeViewMatrix(Pos, tagPos, (0, 0, 1))
        viewMatrix = [round(elem, 2) for elem in viewMatrix]
        projectMatrix = p.computeProjectionMatrixFOV(60, 1, 0.1, 100)     # input: field of view, ratio(width/height),near,far
        projectMatrix = [round(elem, 2) for elem in projectMatrix]
        rgbpix = p.getCameraImage(XSIZE, YSIZE, viewMatrix, projectMatrix)[2]
        return rgbpix[:, :, 0:3]


