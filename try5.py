import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
import gym
import mujoco_py

env = gym.make('Ant-v3')
obs = env.reset()
env.render()
time.sleep(100)