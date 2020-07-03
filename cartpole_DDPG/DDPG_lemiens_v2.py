print("\033[3;36m")

import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import time
import logging
import math
import utils
from utils import *
import Continuous_CartPole
from Continuous_CartPole import *

import tensorboard
tf.compat.v1.disable_eager_execution()

"""
d'après ce papier
https://arxiv.org/pdf/1509.02971v2.pdf
"""


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("policy_dims", "20", "")
flags.DEFINE_string("critic_dims", "", "")

flags.DEFINE_integer("batchs", 1, "")
flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_integer("buffer_size", 10 ** 6, "")

flags.DEFINE_integer("num_iter", 1000, "")
flags.DEFINE_integer("eval_interval", 10,
                     "Evaluate policy without exploration every $n$ "
                     "iterations.")
flags.DEFINE_float("policy_lr", 0.0001, "")
flags.DEFINE_float("critic_lr", 0.00001, "")
flags.DEFINE_float("momentum", 0.9, "")
flags.DEFINE_float("gamma", 0, "")
flags.DEFINE_float("tau", 1, "")

critic_layers = [10,10,6]
actor_layers = [10,10,6]


class DPG(object):

    def __init__(self, obs_size, action_size, q_targets=None, tau=None, name="dpg"):

        # Inputs
        self.obs_size = obs_size
        self.action_size = action_size
        self.q_targets = q_targets
        self.tau = tau
        self.name = name

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._vs = vs

            self._make_inputs()
            self._make_graph()
            self._make_objectives()
            tf.compat.v1.summary.FileWriter("logdir/graphBT", graph=tf.compat.v1.get_default_graph())
            self._make_initialize()
            self._make_updates()

    def _make_inputs(self):
        self.inputs = tf.compat.v1.placeholder(tf.float32, (None, self.obs_size),name="observation")
        self.q_targets = tf.compat.v1.placeholder(tf.float32, (None,), name="q_targets")
        self.tau = self.tau or tf.compat.v1.placeholder(tf.float32, (1,), name="tau")
        self.q_action = tf.compat.v1.placeholder(tf.float32, (None, self.action_size),name="action")

    def _make_graph(self):
        # Build main model: actor
        self.policy = policy_model(self.inputs, self.obs_size, actor_layers, self.action_size, name="policy")
        #self.a_explore = self.noiser(self.inputs, self.a_pred)

        # Build main model: critic (on- and off-policy)
        self.critic = critic_model(self.inputs, self.obs_size, self.policy, self.action_size, critic_layers, name="critic")
        self.critic_off = critic_model(self.inputs, self.obs_size, self.q_action, self.action_size, critic_layers, name="critic", reuse=True)


        # Build tracking models.
        self.policyP = policy_model(self.inputs, self.obs_size, actor_layers, self.action_size, name="policyP")
        self.criticP = critic_model(self.inputs, self.obs_size, self.policyP, self.action_size, critic_layers, name="criticP")


    def _make_objectives(self):
        # TODO: Hacky, will cause clashes if multiple DPG instances.
        # Can't instantiate a VS cleanly either, because policy params might be
        # nested in unpredictable way by subclasses.
        policy_params = [var for var in tf.compat.v1.global_variables() if "policy/" in var.name]
        critic_params = [var for var in tf.compat.v1.global_variables() if "critic/" in var.name]
        self.policy_params = policy_params
        self.critic_params = critic_params

        # Policy objective: maximize on-policy critic activations
        self.policy_objective = -tf.reduce_mean(self.critic,name = "policyObjective")

        # Critic objective: minimize MSE of off-policy Q-value predictions
        q_errors = tf.square(self.q_targets - self.critic_off)
        self.critic_objective = tf.reduce_mean(q_errors, name = "criticObjective")

    def _make_initialize(self):
        with tf.compat.v1.variable_scope("initalize"):
            updates = []
            params = [var for var in tf.compat.v1.global_variables() if "policy/" in var.name]
            track_params = [var for var in tf.compat.v1.global_variables() if "policyP/" in var.name]
            print("\033[4;93m")
            print("Sanity check:")
            print("\033[0;2;93m",end="")
            for param,track_param in zip(params,track_params):
                print("assign : " + track_param.name + "    with : " + param.name)
                update_op = tf.compat.v1.assign(track_param, param)
                updates.append(update_op)

            params = [var for var in tf.compat.v1.global_variables() if "critic/" in var.name]
            track_params = [var for var in tf.compat.v1.global_variables() if "criticP/" in var.name]

            for param,track_param in zip(params,track_params):
                print("assign : " + track_param.name + "    with : " + param.name)
                update_op = tf.compat.v1.assign(track_param, param)
                updates.append(update_op)

            self.init_variables = tf.group(*updates)
        print("\033[0;3;36m",end="")

    def _make_updates(self):
        # Make tracking updates.
        policy_track_update = track_model_updates("%s/policy" % self.name, "%s/policyP" % self.name, self.tau)
        critic_track_update = track_model_updates("%s/critic" % self.name, "%s/criticP" % self.name, self.tau)
        self.track_update = tf.group(policy_track_update, critic_track_update,name="trackUpdate")



def policy_model(input, input_dim, layers_dims, output_dims, name="policy", reuse=None, track_scope=None):
    with tf.compat.v1.variable_scope(name, reuse=reuse, initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5)):
        #return tf.math.l2_normalize(mlp(input, input_dim, output_dims, hidden=layers_dims, track_scope=track_scope))
        #return tf.tanh(mlp(input, input_dim, output_dims, hidden=layers_dims, track_scope=track_scope))
        return mlp(input, input_dim, output_dims, hidden=layers_dims, track_scope=track_scope)




def critic_model(input, input_dim, actions, actions_dim, layers_dims, name="critic", reuse=None, track_scope=None):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        output = mlp(tf.concat([input, actions],axis = 1), input_dim + actions_dim, 1, hidden=layers_dims, bias_output=True, track_scope=track_scope)
        return tf.squeeze(output)


def track_model_updates(main_name, track_name, tau):
    """
    Build an update op to make parameters of a tracking model follow a main model.
    Call outside of the scope of both the main and tracking model.
    Returns:
    A group of `tf.compat.v1.assign` ops which require no inputs (only parameter values).
    """

    print("\033[4;93m")
    print("Sanity check:")
    print("\033[0;2;93m",end="")
    #Attention ca peut facilement casser... on ne check pas que les variable correspondent bien
    updates = []
    params = [var for var in tf.compat.v1.global_variables() if var.name.startswith(main_name + "/")]
    track_params = [var for var in tf.compat.v1.global_variables() if var.name.startswith(track_name + "/")]

    with tf.compat.v1.variable_scope("trackOp"):
        for param,track_param in zip(params,track_params):
            print("track : " + track_param.name + "    with : " + param.name)
            update_op = tf.compat.v1.assign(track_param,tau * param + (1 - tau) * track_param)
            updates.append(update_op)

        print("\033[0;3;36m")
        return tf.group(*updates,)


def play_one(env, dpg, policy_update, critic_update, buffer, exploration = True):
    sess = tf.compat.v1.get_default_session()
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    rewards = []
    costs = []
    #buffer = Buffer()

    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = sess.run(dpg.policy, {dpg.inputs: observation.reshape(1,4)})
        if exploration:
            action += np.random.randn(1) * 0.1
        #print(action.shape)

        prev_observation = observation

        #print(action)
        if(math.isnan(action[0])):
            print("\033[4;91m", end='')
            print("ATTENTION action NaN")
            return 0


        #print("+++++++")
        #print(action.reshape(1,).shape)
        #print("+++++++")

        observation, reward, done, info = env.step(np.clip(action,-1,1))

        buffer.extend(prev_observation,action,reward,observation)
        rewards.append(reward)

        cost = train_batch(dpg, policy_update, critic_update, buffer)
        costs.append(cost)

        if 'visu' in sys.argv:
            env.render()
        iters += 1

    return np.sum(rewards) , np.mean(costs)


def train_batch(dpg, policy_update, critic_update, buffer):
    costs = []
    for i in range(FLAGS.batchs):
        cost = train_mini_batch(dpg, policy_update, critic_update, buffer)
        costs.append(cost)

    return np.mean(costs)

def train_mini_batch(dpg, policy_update, critic_update, buffer):
    sess = tf.compat.v1.get_default_session()

    # Sample a training minibatch.
    try:
        b_states, b_actions, b_rewards, b_states_next = buffer.sample(FLAGS.batch_size)
    except ValueError:
        return 0.0

    # Compute targets (TD error backups) given current Q function.
    qP = sess.run(dpg.criticP, {dpg.inputs: b_states_next})
    b_targets = b_rewards + FLAGS.gamma * qP.flatten()

    # Critic update.
    cost_t, _ = sess.run([dpg.critic_objective, critic_update], {dpg.inputs: b_states, dpg.q_targets: b_targets, dpg.q_action: b_actions})

    # Policy update.
    sess.run(policy_update, {dpg.inputs: b_states})

    #tracking update
    sess.run([dpg.track_update], {dpg.tau: [FLAGS.tau]})

    return cost_t


def build_updates(dpg):

    policy_optim = tf.compat.v1.train.AdamOptimizer(FLAGS.policy_lr)
    #policy_optim = tf.compat.v1.train.GradientDescentOptimizer(FLAGS.policy_lr)
    policy_update = policy_optim.minimize(dpg.policy_objective, var_list=dpg.policy_params)

    #critic_optim = tf.compat.v1.train.MomentumOptimizer(FLAGS.critic_lr, FLAGS.momentum)
    #critic_optim = tf.compat.v1.train.AdamOptimizer(FLAGS.critic_lr)
    critic_optim = tf.compat.v1.train.AdamOptimizer(FLAGS.critic_lr)
    critic_update = critic_optim.minimize(dpg.critic_objective, var_list=dpg.critic_params)

    return policy_update, critic_update



def main():
    tic = time.time()
    env = ContinuousCartPoleEnv()
    obs = env.reset()
    done = False

    D = env.observation_space.shape[0]
    K = env.action_space.shape[0]
    dpg = DPG(D,K)
    tf.compat.v1.summary.FileWriter("logdir/graphAT", graph=tf.compat.v1.get_default_graph())


    policy_update, critic_update = build_updates(dpg)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(dpg.init_variables)

        print("\033[0;1;32m")
        print("===================")
        print("LE DEBUT")
        print("===================")
        #print([var.name for var in tf.compat.v1.global_variables()])

        N = 10000
        totalrewards = np.empty(N)
        costs = np.empty(N)

        buffer = ReplayBuffer(FLAGS.buffer_size,D,K)

        writer = tf.summary.create_file_writer("logdir/plot")
        with writer.as_default():
            for n in range(N):
                explo = not(n % FLAGS.eval_interval == 0)
                totalreward, cost = play_one(env, dpg, policy_update, critic_update, buffer, explo)
                totalrewards[n] = totalreward

                #cost = train_batch(dpg, policy_update, critic_update, buffer)
                costs[n] = cost
                tf.summary.scalar("rewards",totalreward,step = n)
                tf.summary.scalar("costs", cost, step = n)
                writer.flush()

                if n % 50 == 0:
                    print("\033[0;97m", end='')
                    print("Episode: {}                                             ".format(n))
                    print("total reward: {:.5}  avg reward (last 50): {:.5}".format(totalreward,totalrewards[max(0, n-50):(n+1)].mean()))
                    print("cost: {:.5}  avg cost (last 50): {:.5}".format(cost,costs[max(0, n-50):(n+1)].mean()))
                tac = time.time()
                print("\033[3;4;91m", end='')
                print("temps total : {} secondes\r".format(int(tac - tic)), end='')

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)


if __name__ == '__main__':
    main()
