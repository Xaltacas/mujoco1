import gym
import sys
import numpy as np
from ENV.custom_env import *
import compress_json


if __name__ == '__main__':
    env = customEnv()

    buff = {"state":[],
            "action":[],
            "reward":[],
            "done":[],
            "next_state":[]}

    buffer_size = 0
    nb_episode = 0
    while buffer_size < 1000:
        state = env.reset()

        step = 0
        done = False
        while step < 200 and not done and buffer_size < 1000:
            act = state["desired_goal"] - state["achieved_goal"]
            action = [act[0],act[1],act[2],0]
            prev_state = state
            next_state, reward, done, info = env.step(action)

            env.render()

            buff["state"].append(np.concatenate([prev_state["observation"],prev_state["desired_goal"]]).tolist())
            buff["action"].append(action)
            buff["reward"].append(reward)
            buff["done"].append(bool(done))
            buff["next_state"].append(np.concatenate([state["observation"],state["desired_goal"]]).tolist())

            buffer_size +=1
            step +=1
        nb_episode +=1
        print("nb episode : {}     Buffersize : {}".format(nb_episode,buffer_size))

    #print(type(buff))
    #print(type(buff["state"][0]))
    #print(type(buff["action"][0]))
    #print(type(buff["reward"][0]))
    #print(type(buff["next_state"][0]))

    compress_json.local_dump(buff, "preTrain/fetch_small.json.gz")
