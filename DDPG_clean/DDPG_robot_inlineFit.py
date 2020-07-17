"""
test pour pré fit un modèle en scriptant l'action effectué
a pas l'air de marcher...
les papiers disent que montrer que des bonnes action est pas top
donc ca diverge dés qu'on quitte le script et laisse le modèle choisr
"""




import gym
import sys
import time
import compress_json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Utils.noise import OUNoise
from Utils.actor import ActorNetwork
from Utils.critic import CriticNetwork
from Utils.replay_buffer import ReplayBuffer
from ENV.custom_env import *


def train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep):

    if "--save" in sys.argv:
        saver = tf.compat.v1.train.Saver()

    if "--load" in sys.argv:
        print("loading weights")
        loader = tf.compat.v1.train.Saver()
        arg_index = sys.argv.index("--load")
        save_name = sys.argv[arg_index + 1]
        loader.restore(sess,"savedir/"+save_name+"/save")
        print("weights loaded")
    else:
        sess.run(tf.compat.v1.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(buffer_size, 0)

    if "--loadBuff" in sys.argv:
        arg_index = sys.argv.index("--loadBuff")
        buffPath = sys.argv[arg_index + 1]
        print("loading buffer")
        tempBuff = compress_json.local_load("preTrain/"+buffPath+".json.gz")
        nb = buffer_size / len(tempBuff["action"])
        for i in range(int(nb)):
            for s,a,r,d,s1 in zip(tempBuff["state"],tempBuff["action"],tempBuff["reward"],tempBuff["done"],tempBuff["next_state"]):
                replay_buffer.add(np.reshape(s,(actor.s_dim,)),np.reshape(a,(actor.a_dim,)),r,d,np.reshape(s1,(actor.s_dim,)))

        print("buffer loaded")


    max_episodes = ep
    max_steps = 200
    score_list = []
    tcostlist = []

    tic = time.time()

    for i in range(max_episodes):

        state = env.reset()
        state = np.concatenate([state["observation"],state["desired_goal"]])
        score = 0
        cost = 0
        costs = []
        actor_noise.reset()

        if(i % 10 == 0):
            #print("serious:")
            explo=0
        else:
            explo=1

        for j in range(max_steps):

            if '--visu' in sys.argv:
                env.render()

            act = state[-3:] - state[:3]
            act = act if np.linalg.norm(act) == 0 else (act/np.linalg.norm(act))* 0.3
            action = [act[0],act[1],act[2],0]

            action = np.clip(action + actor_noise()*explo*0.1   ,-1,1)

            #print(action)
            next_state, reward, done, info = env.step(action.reshape(4,))


            #print("reward : {:5}    score : {:5}      ".format(reward,score))

            next_state = np.concatenate([next_state["observation"],next_state["desired_goal"]])
            replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                              done, np.reshape(next_state, (actor.s_dim,)))

            state = next_state
            score += reward

            tac = time.time()
            print("\033[0;1;4;97m", end='')
            print("Episode:", end = "")
            print("\033[0;97m", end='')
            print(" {}    ".format(i),end='')
            print("\033[3;4;91m", end='')
            print("temps total : {} secondes\r".format(int(tac - tic)), end='')

            # updating the network in batch
            if replay_buffer.size() < min_batch:
                if done:
                    break
                else:
                    continue

            states, actions, rewards, dones, next_states = replay_buffer.sample_batch(min_batch)
            target_q = critic.predict_target(next_states, actor.predict_target(next_states))

            y = []
            for k in range(min_batch):
                y.append(rewards[k] + critic.gamma * target_q[k] * (1-dones[k]))

            # Update the critic given the targets
            predicted_q_value, _ = critic.train(states, actions, np.reshape(y, (min_batch, 1)))
            cost = y-predicted_q_value
            costs.append(cost)

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(states)
            grads = critic.action_gradients(states, a_outs)
            actor.train(states, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

            if done:
                break

        if i != 0 :
            tcost = np.mean(costs)
            tcostlist.append(tcost)

        score_list.append(score)

        if i % 10 == 0 and i!=0:
            print("\033[0;1;4;97m", end='')
            print("Episode:", end = "")
            print("\033[0;97m", end='')
            print(" {}                                             ".format(i))
            print("{}total reward: {:.5}  avg reward (last 10): {:.5}".format("DONE, " if done else "",score,np.mean(score_list[max(0, i-10):(i+1)])))
            print("cost: {:.5}  avg cost (last 10): {:.5}".format(tcost,np.mean(tcostlist[max(0, i-10):(i+1)])))
            if "--save" in sys.argv:
                arg_index = sys.argv.index("--save")
                save_name = sys.argv[arg_index + 1]
                saver.save(sess, "savedir/" + save_name+"/save")



    print("\033[3;4;91m", end='')
    print("temps total : {} secondes".format(int(tac - tic)))

    return score_list


if __name__ == '__main__':

    with tf.compat.v1.Session() as sess:

        env = customEnv()

        env.seed(0)
        np.random.seed(0)
        tf.compat.v1.set_random_seed(0)

        ep = 10000
        tau = 0.0001
        gamma = 0.99
        min_batch = 32
        actor_lr = 0.00001
        critic_lr = 0.0001
        buffer_size = 1000000
        layers = [1024,512]

        state_dim =  env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor_noise = OUNoise(mu=np.zeros(action_dim))
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, layers, actor_lr, tau, min_batch)
        critic = CriticNetwork(sess, state_dim, action_dim, layers, critic_lr, tau, gamma, actor.get_num_trainable_vars())
        tf.compat.v1.summary.FileWriter("logdir/graphpend", graph=tf.compat.v1.get_default_graph())

        print("\033[0;1;32m")
        print("===================")
        print("LE DEBUT")
        print("===================")

        if "--demo" in sys.argv:
            if "--load" in sys.argv:
                print("loading weights")
                loader = tf.compat.v1.train.Saver()
                arg_index = sys.argv.index("--load")
                save_name = sys.argv[arg_index + 1]
                loader.restore(sess,"savedir/"+save_name+"/save")
                print("weights loaded")
            else:
                print("use --load with --demo")
                quit()

            while True:
                state = env.reset()
                for i in range(200):
                    env.render()
                    action = np.clip(actor.predict(np.reshape(np.concatenate([state["observation"],state["desired_goal"]]), (1, actor.s_dim))),-1,1)
                    next_state, reward, done, info = env.step(np.reshape(action,(4,)))
                    state = next_state
                    if done:
                        break

        else:
            scores = train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep)
            plt.plot([i + 1 for i in range(0, ep, 3)], scores[::3])
            plt.show()
