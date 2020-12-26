import gym
import numpy as np

from matplotlib import pyplot as plt
theta = np.random.uniform(low=-1.5, high=1.5, size=(5,))
theta = np.vstack((theta, np.zeros((5,)))).T.flatten()
print(theta)
print(theta.shape)
