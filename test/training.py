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


tf.compat.v1.disable_eager_execution()
"""
Deuxième version de test pour resoudre fetch
on tente de faire l'optimisation avec cette methode: https://arxiv.org/pdf/1509.02971v2.pdf

implémentations?
https://github.com/hans/rlcomp
"""


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
    self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)


# approximates mu(s)
class ActorModel:
  def __init__(self, D, K, hidden_layer_sizes):
    # create the graph
    # K = number of actions
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    # layer = HiddenLayer(M1, K, lambda x: x, use_bias=False)
    layer = HiddenLayer(M1, K, tf.nn.tanh, use_bias=False)
    self.layers.append(layer)

    # inputs and targets
    self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, D), name='X')
    self.actions = tf.compat.v1.placeholder(tf.float32, shape=(None,4), name='actions')
    self.advantages = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='advantages')

    # calculate output and cost
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    self.predict_op = Z


    selected_probs = tf.math.log(
      tf.reduce_sum(Z * self.actions,axis = 1)
    )

    # self.selected_probs = selected_probs
    cost = tf.reduce_sum(self.advantages * selected_probs)
    self.train_op = tf.compat.v1.train.AdamOptimizer(2.0).minimize(cost)


  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X}) * 0.1

  def sample_action(self, X):
    p = self.predict(X)[0]
    return np.random.choice(len(p), p=p)


# approximates Q(s,a)
class CriticModel:
  def __init__(self, D, K, hidden_layer_sizes):
    # create the graph
    self.layers = []
    M1 = D + K
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, 1, lambda x: x)
    self.layers.append(layer)

    # inputs and targets
    self.S = tf.compat.v1.placeholder(tf.float32, shape=(None, D), name='S') # observation
    self.A = tf.compat.v1.placeholder(tf.float32, shape=(None, K), name='A') # action

    X = tf.concat([self.S,self.A],axis = 1)

    # calculate output and cost
    Z = X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = tf.reshape(Z, [-1]) # the output
    self.predict_op = Y_hat

    self.R = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name='R')
    self.Q = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name='Q')

    Y = R + gamma * self.predict_op

    loss = tf.reduce_sum(Y,tf.math.square(Q))

    self.train_op = tf.compat.v1.train.AdamOptimizer(1e-2).minimize(loss)

  def set_session(self, session):
    self.session = session

  def update(self, S, A, R, Q):
    #X = np.atleast_2d(X)
    #Y = np.atleast_1d(Y)
    self.session.run(self.train_op, feed_dict={self.A: A, self.S: S, self.R: R, self.Q: Q})

  def predict(self, S, A):
    #X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.A: A, self.S: S})

class Buffer:
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.s1 = []
        self.size = 0

    def extend(s,a,r,s1):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s1.append(s1)
        self.size += 1

    def sample(size):

        if len(buffer) < N:
            return None

        idxs = np.random.choice(range(self.size), N, replace=False)
        return self.s[idxs], self.a[idxs], self.r[idxs], self.s1[idxs]



def play_one(env, actor, critic, actor_prime, critic_prime, gamma, training_interval = 10, minibatch_size = 10):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  obs = np.concatenate((observation["observation"],observation["desired_goal"]))

  rewards = []

  buffer = Buffer()

  while not done and iters < 2000:


    action = pmodel.predict(obs).reshape((4,))

    prev_observation = observation
    p_obs = np.concatenate((prev_observation["observation"],prev_observation["desired_goal"]))
    observation, reward, done, info = env.step(action)
    obs = np.concatenate((observation["observation"],observation["desired_goal"]))

    buffer.extend(p_obs,action,reward,obs)
    # update the models
    if iters % training_interval == 0:
        ms, ma, mr, ms1  = buffer.sample(minibatch_size)
        if minibatch != None:
            ap = actor_prime.predict(ms1)
            q = critic.predict(ms,ma)
            critic_prime.update(ms1,ap,mr,q)






    rewards.append(reward)
    if 'visu' in sys.argv:
        env.render()
    iters += 1

  return np.mean(rewards)



def main():
  env = gym.make("custom-v0")
  obs = env.reset()
  done = False

  D = env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0]
  print(env.observation_space)
  K = env.action_space.shape[0]
  pmodel = PolicyModel(D, K, [10])
  vmodel = ValueModel(D, [10])
  init = tf.compat.v1.global_variables_initializer()
  session = tf.compat.v1.InteractiveSession()
  session.run(init)
  pmodel.set_session(session)
  vmodel.set_session(session)
  gamma = 0.99

  print("===================")
  print("LE DEBUT")
  print("===================")

  N = 1000
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    tic = time.time()
    totalreward = play_one(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 5 == 0:
        #print("episode:", n, "mean reward:", totalreward, "avg reward (last 10):", totalrewards[max(0, n-10):(n+1)].mean())
        print("episode: {}  mean reward: {:.5}  avg reward (last 10): {:.5}".format(n,totalreward,totalrewards[max(0, n-10):(n+1)].mean()))
        tac = time.time()
        print("durée : {:.5} secondes".format(tac - tic))

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()
