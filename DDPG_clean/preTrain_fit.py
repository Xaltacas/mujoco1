import gym
import sys
import time
import random
import compress_json
import numpy as np
import tensorflow as tf

from Utils.actor import ActorNetwork
from Utils.critic import CriticNetwork
from ENV.custom_env import *

def sample(buffer,sample_size):
    if (sample_size > len(buffer["action"])):
        print("trop petit buffer")
        quit()

    indxs = np.random.choice(len(buffer["action"]),size=sample_size,replace=False).tolist()

    states =[]
    actions =[]
    rewards =[]

    for i in indxs:
        states.append(buffer["state"][i])
        actions.append(buffer["action"][i])
        rewards.append(buffer["reward"][i])

    return states, actions, rewards



def main():
    with tf.compat.v1.Session() as sess:

        tic = time.time()

        env = customEnv()

        if "--mstep" in sys.argv:
            arg_index = sys.argv.index("--mstep")
            micro_stepping = int(sys.argv[arg_index + 1])
        else:
            micro_stepping = 1

        if "--ep" in sys.argv:
            arg_index = sys.argv.index("--ep")
            ep = int(sys.argv[arg_index + 1])
        else:
            ep = 10000

        tau = 0.001
        gamma = 0.99
        min_batch = 64
        actor_lr = 0.0001
        critic_lr = 0.001
        buffer_size = 1000000
        layers = [1024,512]

        state_dim =  (env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0])*micro_stepping
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, layers, actor_lr, tau, min_batch)
        critic = CriticNetwork(sess, state_dim, action_dim, layers, critic_lr, tau, gamma, actor.get_num_trainable_vars())

        action_wanted = tf.compat.v1.placeholder(tf.float32, (None, action_dim))
        reward_wanted = tf.compat.v1.placeholder(tf.float32, (None, 1))

        actor_target = tf.reduce_mean(tf.square(actor.out-action_wanted))
        critic_target = tf.reduce_mean(tf.square(critic.out-reward_wanted))

        actor_train = tf.compat.v1.train.AdamOptimizer(actor_lr).minimize(actor_target)
        critic_train = tf.compat.v1.train.AdamOptimizer(critic_lr).minimize(critic_target)


        update_target_network_actor = [actor.target_network_params[i].assign(actor.network_params[i]) for i in range(len(actor.target_network_params))]
        update_target_network_critic = [critic.target_network_params[i].assign(critic.network_params[i]) for i in range(len(critic.target_network_params))]

        print("\033[0;1;32m")
        print("===================")
        print("LE DEBUT")
        print("===================")

        print("loading buffer")
        arg_index = sys.argv.index("--loadBuff")
        buffPath = sys.argv[arg_index + 1]
        buffer = compress_json.local_load("preTrain/"+buffPath+".json.gz")
        print("buffer loaded")

        sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()

        i = 0
        while i < ep:
            states, actions, rewards = sample(buffer,min_batch)

            sess.run(actor_train,{actor.inputs: states, action_wanted: actions})
            sess.run(critic_train,{critic.inputs: states, critic.action: actions, reward_wanted: np.reshape(rewards,(min_batch,1))})

            print("\033[0;1;4;97m", end='')
            print("miniBatch {} / {}".format(i,ep), end='')
            print("\033[0;m     ", end='')
            tac = time.time()
            print("\033[3;91m", end='')
            print("{} secondes".format(int(tac - tic)), end='')
            print("\033[0;m                  \r", end='')
            i += 1

        sess.run(update_target_network_actor)
        sess.run(update_target_network_critic)

        arg_index = sys.argv.index("--save")
        save_name = sys.argv[arg_index + 1]
        saver.save(sess, "savedir/" + save_name+"/save")
        print("\033[0;1;32m")
        print("session saved at : " + save_name)


    return 0

if __name__ == "__main__":
    if not("--loadBuff" in sys.argv) or not("--save" in sys.argv):
        print("il faut un buffer et un endroit ou save. à spécifier avec --loadBuff et --save")
        quit()

    main()
