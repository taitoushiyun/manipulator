import gym
import numpy as np
env = gym.make('LunarLanderContinuous-v2')
print(env.observation_space)
print(env.action_space)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.high)
print(env.action_space.low)


a = np.array([-1.5, 0.5])
a = a.clip(env.action_space.low, env.action_space.high)
print(a)