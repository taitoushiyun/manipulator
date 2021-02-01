import re
from collections import deque
import numpy as np
rewards = []
mean_rewards = []
reward_queue = deque(maxlen=10)
with open('data.txt', 'r') as f:
    line = f.readline()
    while line != '':
        if 'reward' in line:
            reward = re.findall(r'reward is (-?\d+\.?\d+)', line)[0]
            reward = float(reward)
            rewards.append(reward)
            reward_queue.append(reward)
            mean_rewards.append(np.mean(reward_queue))
        line = f.readline()
from matplotlib import pyplot as plt

plt.plot(rewards)
plt.plot(mean_rewards)
plt.show()
