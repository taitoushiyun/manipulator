import numpy as np
import time
import torch
import gym
from policies.policy_pi import TanhGaussianPolicy
from ReplayBuffers.simple_sampler import SimpleSampler
from ReplayBuffers.simple_replay_buffer import EnvReplayBuffer
from policies.value_functions import VFunction, QFunction


def training_process():
    torch.manual_seed(1)
    env = gym.make('LunarLanderContinuous-v2')  # Pendulum-v0    LunarLanderContinuous-v2
    policy_pi = TanhGaussianPolicy([64, 64], 8, 2)
    abs_path = '/'.join(str.split(__file__, '/')[:-2]) + '/run_td3/policy_twin' + str(1155) + '.pt'
    policy_pi.load_state_dict(torch.load(abs_path))
    sampler = SimpleSampler(max_path_length=1000, min_pool_size=5000, batch_size=32)
    pooling = EnvReplayBuffer(max_replay_buffer_size=1000000, env=env)

    sampler.initialize(env, policy_pi, pooling)
    num_epochs = 2
    epoch_length = 1000

    for epoch in range(num_epochs):
        print('--------------------------- Epoch %i ---------------------------' % (epoch + 1))
        for t in range(epoch_length):
            sampler.sample(True, False)


training_process()

