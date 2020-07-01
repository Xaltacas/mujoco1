import numpy as np
import gym
import custom_env


#gym.envs.register(
#     id='adcustom-v0',
#     entry_point='custom_env:adcustomEnv',
#     max_episode_steps=1000,
#)
#

env = gym.make("custom-v0")
obs = env.reset()
done = False

print("================")
print(env.observation_space)
print(env.action_space)
print("================")


def policy(observation, desired_goal):

    P = 1
    I = 0.1

    res = [0] * 4

    for i in range(len(desired_goal)):
        res[i] = desired_goal[i] - observation[i]

    res[3] = 0

    return res

iter =0
while not done and iter < 300:
    action = policy(obs['observation'], obs['desired_goal'])
    obs, reward, done, info = env.step(action)
    env.render()

    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(obs['achieved_goal'], substitute_goal, info)
    print('iter {} : reward is {}, substitute_reward is {}'.format(iter,reward, substitute_reward))
    iter += 1

if done:
    print("BRAVO c'est touché")
