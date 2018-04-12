from PYBULLET_ROBOT import PybulletRobot as PR
import time
import numpy as np
import sys

f = open('result.txt', 'w')

pr = PR()
print('simulation starting...')
time.sleep(2)
for i in range(3):
    s = pr._reset()
    ep_reward = 0

    while True:

    	# ending conditions
    	if pr._check_collision():
    		f.write('collison detected, episode again, the reward is: %f \n' %ep_reward)
    		time.sleep(1)
    		break
    	
    	s_, r = pr._step(np.random.normal(-1,1,7))
    	f.write("the next state is: %s the reward is: %f\n" %(s_[0:3], r))
    	ep_reward += r  # aggregate the episode reward

f.close()
