import sys
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent')
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/Algorithms')
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/pytorch_core')
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/policies')
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/ReplayBuffers')
from Algorithms.td3 import TD3
import torch
import numpy as np
import pickle
import gym
import matplotlib.pyplot as plt
from policies.networks import MlpPI
from policies.value_functions import QFunction
from ReplayBuffers.simple_replay_buffer import SimpleReplayBuffer
from ReplayBuffers.simple_sample import SimpleSampler


def training_process():
    act_dims = 2
    obs_dims = 8
    env = gym.make('LunarLanderContinuous-v2')
    policy = MlpPI([100, 100], act_dims, obs_dims)
    print(policy)
    q1_fn = QFunction([100, 100], obs_dims, act_dims)
    print(q1_fn)
    q2_fn = QFunction([100, 100], obs_dims, act_dims)
    sampler = SimpleSampler(max_path_length=500, min_pool_size=2000, batch_size=64)
    base_kwargs = {
        'num_total_epochs': 2000,
        'epoch_length': 2000,
        'num_train_repeat': 1,
        'sampler': sampler,
    }
    pooling = SimpleReplayBuffer(max_replay_buffer_size=int(1e7), observation_dim=obs_dims, action_dim=act_dims)
    td3_agent = TD3(base_kwargs, env, policy, pooling, q1_fn, q2_fn)

    td3_agent.train()
    pickle.dump(sampler.reward_episodes, open('td3_Lunar_learning_process.pkl', 'wb'))
    # print sampler.reward_episodes
    # plt.plot(sampler.reward_episodes)
    # plt.show()


if __name__ == '__main__':
    training_process()





