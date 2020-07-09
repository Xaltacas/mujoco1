"""
honteusement volé ici : https://github.com/shivaverma/OpenAIGym/
"""


import gym
import sys
import time
from ENV.lunarLanderContinuous import LunarLanderContinuous
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Utils.noise import OUNoise
from Utils.actor import ActorNetwork
from Utils.critic import CriticNetwork
from Utils.replay_buffer import ReplayBuffer
import Continuous_CartPole
from Continuous_CartPole import *



def train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep):

    sess.run(tf.compat.v1.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(buffer_size, 0)

    max_episodes = ep
    max_steps = 500
    score_list = []
    tcostlist = []

    tic = time.time()

    for i in range(max_episodes):

        state = env.reset()
        score = 0
        cost = 0
        costs = []

        if(i % 10 == 0):
            #print("serious:")
            explo=0
        else:
            explo=1

        for j in range(max_steps):

            if 'visu' in sys.argv:
                env.render()

            action = np.clip(actor.predict(np.reshape(state, (1, actor.s_dim))) + actor_noise()*explo,-1,1)

            #print(action)
            next_state, reward, done, info = env.step(action[0])
            replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                              done, np.reshape(next_state, (actor.s_dim,)))

            # updating the network in batch
            if replay_buffer.size() < min_batch:
                continue

            states, actions, rewards, dones, next_states = replay_buffer.sample_batch(min_batch)
            target_q = critic.predict_target(next_states, actor.predict_target(next_states))

            y = []
            for k in range(min_batch):
                y.append(rewards[k] + critic.gamma * target_q[k] * (1-dones[k]))

            # Update the critic given the targets
            predicted_q_value, _ = critic.train(states, actions, np.reshape(y, (min_batch, 1)))
            costs.append(y-predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(states)
            grads = critic.action_gradients(states, a_outs)
            actor.train(states, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

            state = next_state
            score += reward

            tac = time.time()
            print("\033[3;4;91m", end='')
            print("temps total : {} secondes\r".format(int(tac - tic)), end='')

            if done:

                break

        tcost = np.mean(costs)
        tcostlist.append(tcost)

        score_list.append(score)

        if i % 10 == 0:
            print("\033[0;1;4;97m", end='')
            print("Episode:", end = "")
            print("\033[0;97m", end='')
            print(" {}                                             ".format(i))
            print("total reward: {:.5}  avg reward (last 10): {:.5}".format(score,np.mean(score_list[max(0, i-10):(i+1)])))
            print("cost: {:.5}  avg cost (last 10): {:.5}".format(tcost,np.mean(tcostlist[max(0, i-10):(i+1)])))

    return score_list


if __name__ == '__main__':

    with tf.compat.v1.Session() as sess:

        env = LunarLanderContinuous()

        env.seed(0)
        np.random.seed(0)
        #tf.set_random_seed(0)

        ep = 10000
        tau = 0.001
        gamma = 0.99
        min_batch = 64
        actor_lr = 0.0001
        critic_lr = 0.001
        buffer_size = 1000000
        layers = [400,300]

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor_noise = OUNoise(mu=np.zeros(action_dim))
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, layers, actor_lr, tau, min_batch)
        critic = CriticNetwork(sess, state_dim, action_dim, layers, critic_lr, tau, gamma, actor.get_num_trainable_vars())
        tf.compat.v1.summary.FileWriter("logdir/graphpend", graph=tf.compat.v1.get_default_graph())

        scores = train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep)
        plt.plot([i + 1 for i in range(0, ep, 3)], scores[::3])
        plt.show()
