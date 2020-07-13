import gym
import sys
import time
import numpy as np
from ENV.custom_env import *
from Utils.noise import OUNoise
import compress_json


if __name__ == '__main__':

    print("\033[0;1;32m")
    print("===================")
    print("Création de donné de train")
    print("===================")

    env = customEnv()

    if "--size" in sys.argv:
        arg_index = sys.argv.index("--size")
        buff_max = int(sys.argv[arg_index + 1])
    else:
        buff_max = 1000

    buff = {"state":[],
            "action":[],
            "reward":[],
            "done":[],
            "next_state":[]}

    actor_noise = OUNoise(mu=np.zeros(4))

    buffer_size = 0
    nb_episode = 0
    tic = time.time()
    while buffer_size < buff_max:
        state = env.reset()

        step = 0
        done = False
        while step < 200 and not done and buffer_size < buff_max:
            act = state["desired_goal"] - state["achieved_goal"]
            act = act if np.linalg.norm(act) == 0 else (act/np.linalg.norm(act))* 0.3
            action = [act[0],act[1],act[2],0]
            prev_state = state
            state, reward, done, info = env.step(action + actor_noise()*0.5)
            #env.render()

            buff["state"].append(np.concatenate([prev_state["observation"],prev_state["desired_goal"]]).tolist())
            buff["action"].append(action)
            buff["reward"].append(reward)
            buff["done"].append(bool(done))
            buff["next_state"].append(np.concatenate([state["observation"],state["desired_goal"]]).tolist())

            buffer_size +=1
            step +=1
            print("\033[0;1;97m", end='')
            print("nb episode : {}     Buffersize : {} / {}   ".format(nb_episode,buffer_size,buff_max), end = '')
            tac = time.time()
            print("\033[3;91m", end='')
            print("{} secondes".format(int(tac - tic)), end='')
            print("\033[0;m                  \r", end='')

        nb_episode +=1

    print("\033[0;1;32m")
    print("sauvegarde du buffer...")
    if "--save" in sys.argv:
        arg_index = sys.argv.index("--save")
        save_name = sys.argv[arg_index + 1]
        compress_json.local_dump(buff, "preTrain/"+ save_name + ".json.gz")
        print('sauvegardé en tant que "'+ save_name+'"')
    else:
        compress_json.local_dump(buff, "preTrain/default.json.gz")
        print('sauvegardé en tant que "default"')
