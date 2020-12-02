from PPO.ppo_agent import PPO_agent, ReplayBuffer
from PPO.actor_critic import Actor_critic

import torch
import matplotlib.pyplot as plt
from vrep_con.vrep_utils import ManipulatorEnv


def training_process():
    env = ManipulatorEnv(0)
    obs_dims = env.observation_space.shape[0]
    act_dims = env.action_space.shape[0]
    print(f'obs_dims is {obs_dims}')
    print(f'act_dims is {act_dims}')
    buffer_config = {
        'buffer_size': 256,
        'batch_size': 32,
    }
    ppo_config = {
        'num_episodes': 10000,
        'max_length_per_episode': 100,
        'ppo_epoch': 4,
        'clip_epsilon': 0.2,
        'gamma': 0.99,
        'weight_epsilon': 0.000,
        'lr': 3e-4,
    }
    actor_config = {
        'hidden_sizes': [64, 64],
        'hidden_activation': torch.tanh,
    }
    critic_config = {
        'hidden_sizes': [64, 64],
        'hidden_activation': torch.tanh
    }
    pooling = ReplayBuffer(buffer_size=buffer_config['buffer_size'],
                           act_dims=act_dims, obs_dims=obs_dims,
                           batch_size=buffer_config['batch_size'])
    actor_critic = Actor_critic(actor_obs_dims=obs_dims, actor_hidden_sizes=actor_config['hidden_sizes'],
                                actor_action_dims=act_dims, critic_obs_dims=obs_dims,
                                critic_hidden_sizes=critic_config['hidden_sizes'])

    ppo = PPO_agent(env, actor_critic, num_episodes=ppo_config['num_episodes'],
                    max_steps_per_episodes=ppo_config['max_length_per_episode'], pooling=pooling,
                    clip_epsilon=ppo_config['clip_epsilon'], gamma=ppo_config['gamma'], lr=ppo_config['lr'],
                    ppo_epoch=ppo_config['ppo_epoch'], weight_epsilon=ppo_config['weight_epsilon'])
    ppo.train()
    # plt.plot(ppo.rewards_learning_prcoess)
    # plt.show()


if __name__ == '__main__':
    training_process()
