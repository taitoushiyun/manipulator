import numpy as np
import gym

env = gym.make('MountainCarContinuous-v0')
env.reset()
step = 0
while True:
    obs, reward, done, info = env.step(env.action_space.sample())
    # env.render()
    step += 1
    if done:
        print(f'******************************{step}**************************************')
        env.reset()
        step = 0
    # else:
    #     print(step)