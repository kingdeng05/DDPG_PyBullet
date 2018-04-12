# this script defines the actor class for the DDPG robot class

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

BATCH_SIZE = 32

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter, S, S_, A):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.S = S
        self.S_ = S_
        self.A = A

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(self.S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(self.S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s, a):   # batch update
        self.sess.run(self.train_op, feed_dict={self.S: s, self.A: a})
        # the following method for soft replace target params is computational expansive
        # target_params = (1-tau) * target_params + tau * eval_params
        # self.sess.run([tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(self.t_params, self.e_params)])

        # instead of above method, I use a hard replacement here
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={self.S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # self.a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr / BATCH_SIZE)  # (- learning rate) for ascent policy, div to take mean
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))