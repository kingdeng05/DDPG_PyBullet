import math
import os
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p


def takepicture(Pos, Orn):
    # Pos is the position of end effect, orn is the orientation of the end effect
    rotmatrix = p.getMatrixFromQuaternion(Orn)
    # distance from camera to focus
    distance = 0.2
    # where does camera aim at
    camview = list()
    for i in range(3):
        camview.append(np.dot(rotmatrix[i * 3:i * 3 + 3], (0, 0, distance)))
    # camview[2] = camview[2]
    tagPos = np.add(camview, Pos)

    # p.removeBody(kukaId)
    viewMatrix = p.computeViewMatrix(Pos, tagPos, (0, 0, 1))
    # viewMatrix=p.computeViewMatrix([-2, 0, 2], [0, 0, 1],[0, 0, 1])
    projectMatrix = p.computeProjectionMatrixFOV(60, 1, 0.1, 100)  # input: field of view, ratio(width/height),near,far
    rgbpix = p.getCameraImage(128, 128, viewMatrix, projectMatrix)[2]

    return rgbpix[:, :, 0:3]


path = os.getcwd()
p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -9.8)
kukaId = p.loadURDF(path + "/kuka_lwr/kuka.urdf", (0, 0, 0), useFixedBase=True)
groundId = p.loadURDF(path + "/floor/plane100.urdf", (0, 0, 0), useFixedBase=True)
numjoint = p.getNumJoints(kukaId)

initpos = (0.5, 0, 0.9)
initorn = ([0, math.pi / 2, 0])

jd = [10, 10, 10, 10, 1, 1, 0.1]
jointinit = [0, -0.488, 0, 0.307, 0, -0.694, 0]
# set initial pos for robot
for j in range(500):
    p.stepSimulation()
    for i in range(numjoint):
        # inverse kinematic
        # jointinit = p.calculateInverseKinematics(kukaId, 6, initpos, initorn, jointDamping=jd)
        # p.setJointMotorControl2(bodyIndex = kukaId, jointIndex = i, controlMode = p.POSITION_CONTROL, targetPosition = jointinit[i], targetVelocity = 0, force = 500, positionGain = 0.03, velocityGain = 1)
        p.resetJointState(kukaId, i, jointinit[i])

# position and orientation of end effector
endPos, endOrn = p.getLinkState(kukaId, 6)[0:2]
rotmatrix = p.getMatrixFromQuaternion(endOrn)
# distance from camera to focus

# setup the ball after the robot get to its initial position
ballId = p.loadURDF(path + "/ball/sphere_small.urdf", (1.5, 0, 0.9), useFixedBase=False)
rgbpix = takepicture(Pos=endPos, Orn=endOrn)

rgimgplot = plt.imshow(np.reshape(np.array(rgbpix) / 255.0, (128, 128, 3)))
plt.show()


