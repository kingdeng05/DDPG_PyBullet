"""
ECE 285 Final Project - Robot Learn Physics

---- Team Member ------
Jingyi Yang, Siwei Guo, Maxim Jiao, Fuheng Deng

----- Project ---------
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning
DDPG is Actor Critic based algorithm
set up 6-link kuka arm to learn ball's free fall physics

----- Requirements ----
python 3.5
tensorflow 1.1.0
gym 0.9.1
numpy 1.12

----- Credits To ------
reference code from mofanzhou's DDPG github folder: 
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG
million thanks to the contribution by mofan zhou

"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from ACTOR import Actor
from CRITIC import Critic
from MEMORY import Memory
from PYBULLET_ROBOT import *

np.random.seed(1)
tf.set_random_seed(1)

 ####################   SET UP PARAMETERS ##################################

MAX_EPISODES = 10000
#MAX_EP_STEPS = 400
LR_A = 0.01  # learning rate for actor
LR_C = 0.01  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # Soft update for target param, but this is computationally expansive
# so we use replace_iter instead
REPLACE_ITER_A = 500 
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 2000
BATCH_SIZE = 20
#######################INITIALIZE PYBULLET INSTANCE#########################

f = open('result.txt','w')
# choose mode based upon user input
if len(sys.argv) is 1:
    pr = PybulletRobot(ff=f)
elif sys.argv[1] is 'D':
    pr = PybulletRobot('DIRECT',f)
elif sys.argv[1] is 'G':
    pr = PybulletRobot('GUI',f)
else:
    print('No such mode exist...exit..')
    exit()

state_dim = pr._state_space_dim()
action_dim = pr._action_space_dim()

######################  SETUP TENSORFLOW  ##################################

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('A'):
    A = tf.placeholder(tf.float32, shape=[None, action_dim], name='a')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

# create a session
sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
action_bound = 1
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACE_ITER_A, S, S_, A)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACE_ITER_C, actor.a_, S, A, S_, R)
M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)  # replay buffer store sequence: s, a, r, s_
actor.add_grad_to_graph(critic.a_grads)

###################### program starts #####################################

# initialize global variable
sess.run(tf.global_variables_initializer())
cnt_epi = 0
# control exploration randomness
var = 2

#savera = tf.train.import_meta_graph('./actor_checkpoint/actor.meta')
#saverc = tf.train.import_meta_graph('./critic_checkpoint/critic.meta')
#savera.restore(actor.sess, tf.train.latest_checkpoint('./actor_checkpoint'))
#saverc.restore(critic.sess, tf.train.latest_checkpoint('./critic_checkpoint'))

# start iteration through given number of episodes
for i in range(int(MAX_EPISODES/50)):
    for j in range(50):
        s = pr._reset()
        ep_reward = 0
        cnt  = 0
        while True:
            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), -action_bound, action_bound)    # add randomness to action selection for exploration
            s_, r, center = pr._step(a)                 # the step (or act) function is predefined in gym (next state and r can be calculated via this)
            M.store_transition(s, a, r / 10, s_)
            if M.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                #  the sequence in sample matrix is state, action, reward, state_
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                # learn from the batch
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s, b_a)

            s = s_
            ep_reward += r  # aggregate the episode reward
            cnt += 1

            
            # ending conditions
            if center[0] < 0:
                f.write('Episode: %i, Reward: %i, Explore: %.2f \n' % ((i+1)*(j+1), int(ep_reward), var))
                f.write('number of iteration is %i \n' % cnt)
                print('number of iteration is %i, reward is: %d' %(cnt, int(ep_reward)))
                break
            '''
            if pr._check_collision():
                f.write('Episode: %i, Reward: %i, Explore: %.2f \n' % ((i+1)*(j+1), int(ep_reward), var))
                f.write('number of iteration is %i \n' % cnt)
                print(int(ep_reward))
                break
            '''

        cnt_epi += 1
        print('episode %i' % cnt_epi)

    savera = tf.train.Saver() # save every vector
    saverc = tf.train.Saver() # save every vector
    savera.save(actor.sess, "./actor_checkpoint/actor")
    saverc.save(critic.sess, "./critic_checkpoint/critic")
        



