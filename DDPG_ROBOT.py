"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
6 link robot example

Using:
python 3.5.3
tensorflow 1.1.0
gym 0.9.1
numpy 1.12

reference code from mofanzhou's DDPG github folder: 
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG
million thanks to the contribution by mofan zhou

"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import gym
from ACTOR import Actor
from CRITIC import Critic
from MEMORY import Memory

np.random.seed(1)
tf.set_random_seed(1)

 ########################SET UP PARAMETERS##################################

MAX_EPISODES = 5000
MAX_EP_STEPS = 400
LR_A = 0.01  # learning rate for actor
LR_C = 0.01  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # Soft update for target param, but this is computationally expansive
# so we use replace_iter instead
REPLACE_ITER_A = 500 
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 7000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = False
ENV_NAME = 'Pendulum-v0'

########################START TRAINING#######################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

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
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACE_ITER_A, S, S_, A)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACE_ITER_C, actor.a_, S, A, S_, R)
actor.add_grad_to_graph(critic.a_grads)

# initialize global variable
sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

var = 3  # control exploration

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        # Added exploration noise
        a = actor.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)                 # the step (or act) function is predefined in gym (next state and r can be calculated via this)

        M.store_transition(s, a, r / 10, s_)

        if M.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            #  the sequence in sample matrix is state, action, reward, state_
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s, b_a)

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -1000:
                RENDER = True
            break