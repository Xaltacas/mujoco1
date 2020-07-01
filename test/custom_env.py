import os
import gym
from gym import utils

import custom_fetchEnv


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class customEnv(custom_fetchEnv.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="ok"):
        #position de la base du robot
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        custom_fetchEnv.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.02,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


gym.envs.register(
     id='custom-v0',
     entry_point='custom_env:customEnv',
     max_episode_steps=200,
)
