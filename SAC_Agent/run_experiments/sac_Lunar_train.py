import numpy as np
import time
import torch
import gym
from SAC_Agent.policies.policy_pi import TanhGaussianPolicy
from SAC_Agent.ReplayBuffers.simple_sampler import SimpleSampler
from SAC_Agent.ReplayBuffers.simple_replay_buffer import EnvReplayBuffer
from SAC_Agent.policies.value_functions import VFunction, QFunction
from SAC_Agent.algorithms.twin_sac_algorithm import SAC
import matplotlib.pyplot as plt
import pickle


torch.manual_seed(3)


def training_process():
    start_time = time.time()
    env = gym.make('LunarLanderContinuous-v2')             #     Pendulum-v0    LunarLanderContinuous-v2
    policy_pi = TanhGaussianPolicy([64, 64], 8, 2)
    sampler = SimpleSampler(max_path_length=500, min_pool_size=2000, batch_size=100)               #  batch_size=32
    pooling = EnvReplayBuffer(max_replay_buffer_size=1000000, env=env)

    Q1_fn = QFunction([64, 64], 1, 10)
    Q2_fn = QFunction([64, 64], 1, 10)
    V_fn = VFunction([64, 64], 1, 8)
    base_kwargs = {
        'num_total_epochs': 2000,
        'epoch_length': 2000,
        'num_train_repeat': 1,
        'sampler': sampler
    }
    sac_agent = SAC(base_kwargs, env, policy_pi, pooling, Q1_fn, Q2_fn, V_fn, use_automatic_entropy_tuning=False,
                    policy_update_period=2)               # policy_update_period=5, target_update_period=5
    sac_agent.train()

    print ('Total training time:   %s[s]' % str(time.time() - start_time))
    # pickle.dump(sampler.reward_episodes, open('training_reward_process.pkl', 'wb'))
    # num_epochs = 5
    # epoch_length = 500
    # for epoch in range(num_epochs):
    #     print('--------------------------- Epoch %i ---------------------------' % (epoch + 1))
    #     for t in range(epoch_length):
    #         sampler.sample(True)
    pickle.dump(sampler.reward_episodes, open('sac_Lunar_learning_process.pkl', 'wb'))
    # print sampler.reward_episodes
    plt.plot(sampler.reward_episodes)
    plt.show()
    # s = pickle.load(open('sac_Lunar_learning_process.pkl', 'rb'))
    # print s


training_process()
# test1()
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = float(env.action_space.high[0])











