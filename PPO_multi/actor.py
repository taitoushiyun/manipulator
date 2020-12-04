import numpy as np
from abc import ABC, abstractmethod
import torch


class AbstractEnvWorker(ABC):
    def __init__(self, *, env, policy):
        self.env = env
        self.policy = policy
        self.nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = None

    @abstractmethod
    def run(self):
        raise NotImplementedError


class ReplayBuffer(object):
    def __init__(self, buffer_size, act_dims, obs_dims):
        self.buffer_size = buffer_size
        self.observations = np.zeros([buffer_size, obs_dims])
        self.actions = np.zeros([buffer_size, act_dims])
        self.rewards = np.zeros([buffer_size, 1])
        self.old_log_probs = np.zeros([buffer_size, 1])
        self.values = np.zeros([buffer_size, 1])
        self.dones = np.zeros([buffer_size, 1])
        self.cur_index = 0

    def store_data(self, cur_obs, cur_action, reward, done, old_log_prob, value):
        self.observations[self.cur_index] = cur_obs
        self.actions[self.cur_index] = cur_action
        self.rewards[self.cur_index] = reward
        self.old_log_probs[self.cur_index] = old_log_prob
        self.dones[self.cur_index] = done
        self.values[self.cur_index] = value
        self.cur_index += 1

    def clear_data(self):
        self.cur_index = 0

    @property
    def enough_data(self):
        return self.cur_index == self.buffer_size


class SamplerWorker(AbstractEnvWorker):
    def __init__(self, *, env, policy, gamma, lammbda, batch_size, act_dim, obs_dim,
                 reward_record, reward_record_size, path_len_record, path_len_record_size, max_episode_steps,
                 num_episodes, num_episodes_lock):
        super(SamplerWorker, self).__init__(env=env, policy=policy)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lammbda = lammbda
        self.pooling = ReplayBuffer(batch_size, act_dim, obs_dim)
        self.reward_record = reward_record
        self.reward_record_size = reward_record_size
        self.path_len_record = path_len_record
        self.path_len_record_size = path_len_record_size
        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes
        self.num_episodes_lock = num_episodes_lock
        self.episode_reward = 0.
        self.path_len = 0

    def run(self):
        if self.obs is None:
            self.obs = cur_obs = self.env.reset()
        else:
            cur_obs = self.obs
        for i in range(self.batch_size):
            action, old_log_prob, value = self.policy.select_action(torch.tensor(cur_obs[None], dtype=torch.float))
            next_obs, reward, done, infos = self.env.step(action)
            self.pooling.store_data(cur_obs, action, reward, done, old_log_prob, value)
            self.episode_reward += reward
            self.path_len += 1
            self.obs = cur_obs = next_obs
            if done or self.path_len == self.max_episode_steps:
                with self.num_episodes_lock:
                    self.num_episodes.value += 1
                print("Episode: %d,          Path length: %d       Reward: %f"
                      % (self.num_episodes.value, self.path_len, self.episode_reward))
                if len(self.reward_record) > self.reward_record_size:
                    self.reward_record.pop(0)
                self.reward_record.append(self.episode_reward)
                if len(self.path_len_record) > self.path_len_record_size:
                    self.path_len_record.pop(0)
                self.path_len_record.append(self.path_len)
                self.episode_reward = 0
                self.path_len = 0
                self.obs = cur_obs = self.env.reset()
        observations, actions = self.pooling.observations, self.pooling.actions
        rewards, dones = self.pooling.rewards, self.pooling.dones
        values, old_log_probs = self.pooling.values, self.pooling.old_log_probs
        self.pooling.clear_data()
        returns = self.compute_gae(next_obs, rewards, values, dones)
        return returns, values,  old_log_probs, observations, actions

    def compute_gae(self, next_obs, rewards, values, dones):
        _, _, value_t_1 = self.policy.select_action(torch.tensor(next_obs[None], dtype=torch.float))
        gae = 0
        returns = np.zeros_like(values)
        for step in reversed(range(rewards.shape[0])):
            td_delta = rewards[step] + self.gamma * (1. - dones[step]) * value_t_1 - values[step]
            value_t_1 = values[step]
            gae = self.gamma * self.lammbda * (1. - dones[step]) * gae + td_delta
            returns[step] = gae + values[step]
        return returns


class EvalWorker(AbstractEnvWorker):
    """
    return : (batch*steps, H, W, C), (batch*steps), (batch*steps), (batch*steps), (batch*steps)
    """
    def __init__(self, *, env, policy, eval_episodes):
        super(EvalWorker, self).__init__(env=env, policy=policy)
        self.eval_episodes = eval_episodes

    def run(self):
        total_reward = 0.0
        for episode in range(self.eval_episodes):
            self.obs = cur_obs = self.env.reset()
            while True:
                action, _, _ = self.policy.select_action(torch.tensor(cur_obs[None], dtype=torch.float))
                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                cur_obs = next_obs
                if done:
                    break
        mean_reward = total_reward / self.eval_episodes
        return mean_reward




