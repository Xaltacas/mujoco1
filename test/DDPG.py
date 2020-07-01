print("\033[3;37m", end='')

import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import custom_env
import time
import logging
import math

import tensorboard
#from tensorflow import keras


tf.compat.v1.disable_eager_execution()

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("policy_dims", "20", "")
flags.DEFINE_string("critic_dims", "", "")

flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_integer("buffer_size", 10 ** 6, "")

flags.DEFINE_integer("num_iter", 1000, "")
flags.DEFINE_integer("eval_interval", 10,
                     "Evaluate policy without exploration every $n$ "
                     "iterations.")
flags.DEFINE_float("policy_lr", 0.001, "")
flags.DEFINE_float("critic_lr", 0.0001, "")
flags.DEFINE_float("momentum", 0.9, "")
flags.DEFINE_float("gamma", 0.95, "")
flags.DEFINE_float("tau", 0.01, "")


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
      self._make_updates()

  def _make_inputs(self):
    self.inputs = (tf.compat.v1.placeholder(tf.float32, (None, self.obs_size),name="inputs"))
    self.q_targets = (tf.compat.v1.placeholder(tf.float32, (None,), name="q_targets"))
    self.tau = self.tau or tf.compat.v1.placeholder(tf.float32, (1,), name="tau")

  def _make_graph(self):
    # Build main model: actor
    self.policy = policy_model(self.inputs, self.obs_size, [10], self.action_size, name="policy")
    #self.a_explore = self.noiser(self.inputs, self.a_pred)

    # Build main model: critic (on- and off-policy)
    self.critic = critic_model(self.inputs, self.obs_size, self.policy, self.action_size, [6], name="critic")
    #self.critic_off = critic_model(self.inputs, self.a_explore, self.mdp_spec,self.spec, name="critic", reuse=True)


    # Build tracking models.
    self.policyP = policy_model(self.inputs, self.obs_size, [10], self.action_size, name="policyP")
    self.criticP = critic_model(self.inputs, self.obs_size, self.policyP, self.action_size, [6], name="criticP")

    #print("=======")
    updates = []
    params = [var for var in tf.compat.v1.global_variables()
              if "policy/" in var.name]
    track_params = [var for var in tf.compat.v1.global_variables()
                      if "policyP/" in var.name]

    for param,track_param in zip(params,track_params):
        #print("assign : " + track_param.name + "    with : " + param.name)
        update_op = tf.compat.v1.assign(track_param, param)
        updates.append(update_op)

    params = [var for var in tf.compat.v1.global_variables()
              if "critic/" in var.name]
    track_params = [var for var in tf.compat.v1.global_variables()
                      if "criticP/" in var.name]

    for param,track_param in zip(params,track_params):
        #print("assign : " + track_param.name + "    with : " + param.name)
        update_op = tf.compat.v1.assign(track_param, param)
        updates.append(update_op)

    self.init_variables = tf.group(*updates)

    #print("=======")


  def _make_objectives(self):
    # TODO: Hacky, will cause clashes if multiple DPG instances.
    # Can't instantiate a VS cleanly either, because policy params might be
    # nested in unpredictable way by subclasses.
    policy_params = [var for var in tf.compat.v1.global_variables()
                     if "policy/" in var.name]
    critic_params = [var for var in tf.compat.v1.global_variables()
                     if "critic/" in var.name]
    self.policy_params = policy_params
    self.critic_params = critic_params

    # Policy objective: maximize on-policy critic activations
    self.policy_objective = -tf.reduce_mean(self.critic)

    # Critic objective: minimize MSE of off-policy Q-value predictions
    q_errors = tf.square(self.q_targets - self.critic)
    self.critic_objective = tf.reduce_mean(q_errors)


  def _make_updates(self):
    # Make tracking updates.
    policy_track_update = track_model_updates("%s/policy" % self.name, "%s/policyP" % self.name, self.tau)
    critic_track_update = track_model_updates("%s/critic" % self.name, "%s/criticP" % self.name, self.tau)
    self.track_update = tf.group(policy_track_update, critic_track_update)

    # SGD updates are left to client.

class Buffer:
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.s1 = []
        self.size = 0

    def extend(self,s,a,r,s1):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s1.append(s1)
        self.size += 1

    def sample(self,size):

        if self.size < size:
            raise ValueError("Not enough examples in buffer (just %i) to fill a batch of %i."
               % (self.size, size))

        idxs = np.random.choice(range(self.size), size, replace=False)
        #print("======")
        #print(idxs)
        #print("======")
        idxs = idxs.astype(int)
        return np.squeeze(np.array(self.s)[idxs]), np.squeeze(np.array(self.a)[idxs]), np.squeeze(np.array(self.r)[idxs]), np.squeeze(np.array(self.s1)[idxs])

#def match_variable(name, scope_name):
#  """
#  Match a variable (initialize with same value) from another variable scope.
#  After initialization, the values of the two variables are not tied in any
#  way.
#  """
#
#
#  with tf.compat.v1.variable_scope(scope_name):
#    track_var = tf.compat.v1.get_variable(name)
#
#  # Create a dummy initializer.
#  initializer = lambda *args, **kwargs: track_var.initialized_value()
#
#  return tf.compat.v1.get_variable(name, shape=track_var.get_shape(),
#                         initializer=initializer)
#
def mlp(inp, inp_dim, outp_dim, track_scope=None, hidden=None, f=tf.tanh, bias_output=False):
  if not hidden:
    hidden = []

  layer_dims = [inp_dim] + hidden + [outp_dim]
  x = inp

  for i, (src_dim, tgt_dim) in enumerate(zip(layer_dims, layer_dims[1:])):
    Wi_name, bi_name = "W" +str(i), "b"+str(i)

    #Wi = ((track_scope and match_variable(Wi_name, track_scope))
    #      or tf.compat.v1.get_variable("W%i" % i, (src_dim, tgt_dim)))
    Wi = tf.Variable(tf.random.normal(shape=(src_dim, tgt_dim)),name = Wi_name)
    x = tf.matmul(x, Wi)

    final_layer = i == len(layer_dims) - 2
    if not final_layer or bias_output:
    #  bi = ((track_scope and match_variable(bi_name, track_scope))
    #        or tf.compat.v1.get_variable("b%i" % i, (tgt_dim,),
    #                           initializer=tf.zeros_initializer))
      bi = tf.Variable(np.zeros(shape =(tgt_dim,)).astype(np.float32),name = bi_name)
      x += bi

    if not final_layer:
      x = f(x)

  return x


def policy_model(input, input_dim, layers_dims, output_dims, name="policy", reuse=None, track_scope=None):

  with tf.compat.v1.variable_scope(name, reuse=reuse, initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5)):
    return mlp(input, input_dim, output_dims, hidden=layers_dims, track_scope=track_scope)


def critic_model(input, input_dim, actions, actions_dim, layers_dims, name="critic", reuse=None, track_scope=None):

  with tf.compat.v1.variable_scope(name, reuse=reuse):
      output = mlp(tf.concat([input, actions],axis = 1), input_dim + actions_dim, 1, hidden=layers_dims, bias_output=True, track_scope=track_scope)
      return tf.squeeze(output)

def track_model_updates(main_name, track_name, tau):
  """
  Build an update op to make parameters of a tracking model follow a main model.
  Call outside of the scope of both the main and tracking model.
  Returns:
  A group of `tf.compat.v1.assign` ops which require no inputs (only parameter values).
  """
  updates = []
  params = [var for var in tf.compat.v1.global_variables()
            if var.name.startswith(main_name + "/")]
  track_params = [var for var in tf.compat.v1.global_variables()
                    if var.name.startswith(track_name + "/")]

  for param,track_param in zip(params,track_params):
      #track_param_name = param.op.name.replace(main_name + "/",track_name + "/")
      #track_param_name += ":0"
      #try:
      #  track_param = tf.compat.v1.get_variable(track_param_name)
      #except ValueError:
      #  logging.warning("Tracking model variable %s does not exist",track_param_name)
      #  continue

      #print("updating : " + track_param.name + "    with : " + param.name)
      update_op = tf.compat.v1.assign(track_param,tau * param + (1 - tau) * track_param)
      updates.append(update_op)

  return tf.group(*updates)

def play_one(env, dpg, policy_update, critic_update, gamma, exploration = True):
    sess = tf.compat.v1.get_default_session()
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    obs = np.concatenate((observation["achieved_goal"],observation["desired_goal"])).reshape(1,6)
    rewards = []
    buffer = Buffer()


    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = sess.run(dpg.policyP, {dpg.inputs: obs}).reshape(4,)
        if exploration:
            action = action * 0.1 + np.random.randn(4)
        #print(action.shape)

        prev_observation = observation
        p_obs = np.concatenate((prev_observation["achieved_goal"],prev_observation["desired_goal"])).reshape(1,6)

        #print(action)
        if(math.isnan(action[0])):
            return 0
        observation, reward, done, info = env.step(np.clip(action,-1,1))

        obs = np.concatenate((observation["achieved_goal"],observation["desired_goal"])).reshape(1,6)
        buffer.extend(p_obs,action,reward,obs)

        cost = train_batch(dpg, policy_update, critic_update, buffer)


        rewards.append(reward)
        if 'visu' in sys.argv:
            env.render()
        iters += 1

    return np.mean(rewards)

def train_batch(dpg, policy_update, critic_update, buffer):
  sess = tf.compat.v1.get_default_session()

  # Sample a training minibatch.


  try:
      b_states, b_actions, b_rewards, b_states_next = buffer.sample(FLAGS.batch_size)
  except ValueError:
    return 0.0

  # Compute targets (TD error backups) given current Q function.
  qP = sess.run(dpg.criticP, {dpg.inputs: b_states_next})
  b_targets = b_rewards + FLAGS.gamma * qP.flatten()

  # Policy update.
  sess.run(policy_update, {dpg.inputs: b_states})

  # Critic update.
  cost_t, _ = sess.run([dpg.critic_objective, critic_update], {dpg.inputs: b_states, dpg.q_targets: b_targets})
  sess.run([dpg.track_update], {dpg.tau: [FLAGS.tau]})

  return cost_t

def build_updates(dpg):

    #policy_optim = tf.compat.v1.train.MomentumOptimizer(FLAGS.policy_lr, FLAGS.momentum)
    policy_optim = tf.compat.v1.train.GradientDescentOptimizer(0.02)
    policy_update = policy_optim.minimize(dpg.policy_objective, var_list=dpg.policy_params)

    #critic_optim = tf.compat.v1.train.MomentumOptimizer(FLAGS.critic_lr, FLAGS.momentum)
    critic_optim = tf.compat.v1.train.AdamOptimizer(0.02)
    critic_update = critic_optim.minimize(dpg.critic_objective, var_list=dpg.critic_params)

    return policy_update, critic_update


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def main():
    tic = time.time()
    env = gym.make("custom-v0")
    obs = env.reset()
    done = False

    D = env.observation_space["achieved_goal"].shape[0] + env.observation_space["desired_goal"].shape[0]
    K = env.action_space.shape[0]
    dpg = DPG(D,K)
    #tf.compat.v1.summary.FileWriter("logdir", graph=tf.compat.v1.get_default_graph())
    policy_update, critic_update = build_updates(dpg)


    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(dpg.init_variables)

        print("\033[0;1;32m")
        print("===================")
        print("LE DEBUT")
        print("===================")

        N = 1000
        totalrewards = np.empty(N)
        costs = np.empty(N)
        for n in range(N):
            #tic = time.time()
            if n % 10 == 0:
                explo = False
            else:
                explo = True

            totalreward = play_one(env, dpg, policy_update, critic_update, FLAGS.gamma,explo)
            totalrewards[n] = totalreward
            if n % 10 == 0:
                #print("episode:", n, "mean reward:", totalreward, "avg reward (last 10):", totalrewards[max(0, n-10):(n+1)].mean())
                print("\033[0;97m", end='')
                print("Episode: {}  mean reward: {:.5}  avg reward (last 10): {:.5}".format(n,totalreward,totalrewards[max(0, n-10):(n+1)].mean()))
                tac = time.time()
                print("\033[3;4;91m", end='')
                print("temps total : {:.5} secondes".format(tac - tic))



    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()
