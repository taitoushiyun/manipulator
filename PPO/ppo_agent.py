import torch
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import visdom
from PPO.logger import logger


class ReplayBuffer(object):
    def __init__(self, buffer_size, act_dims, obs_dims, batch_size=32):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.observations, self.actions, self.rewards = np.zeros([buffer_size, obs_dims]), \
                                                        np.zeros([buffer_size, act_dims]), np.zeros([buffer_size, 1])
        self.old_log_probs, self.values, self.dones = np.zeros([buffer_size, 1]), np.zeros([buffer_size, 1]), \
                                                      np.zeros([buffer_size, 1])
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


class PPO_agent(object):
    def __init__(self, args, env, actor_critic, num_episodes, max_steps_per_episodes, pooling,
                 clip_epsilon=0.2, gamma=0.99, lr=3e-4, ppo_epoch=4, weight_epsilon=0.001):
        self.args = args
        self.env, self.actor_critic = env, actor_critic
        self.num_episodes, self.max_steps_per_episodes, self.pooling = num_episodes, max_steps_per_episodes, pooling
        self.rewards_learning_prcoess = []

        self.clip_epsilon, self.gamma, self.ppo_epoch, self.weight_epsilon = clip_epsilon, gamma, ppo_epoch, weight_epsilon
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.iter_steps = 0
        import os
        os.makedirs('checkpoints', exist_ok=True)
        self.vis = visdom.Visdom(port=self.args.visdom_port, env=self.args.code_version)

        self.reward_cnt = []

    def train(self):
        for episode_t in range(self.num_episodes):
            cur_obs = self.env.reset()
            path_length, path_rewards = 0, 0.
            while True:
                path_length += 1
                action, old_log_prob, value = self.actor_critic.select_action(torch.FloatTensor(cur_obs[None]))
                next_obs, reward, done, _ = self.env.step(action)
                self.pooling.store_data(cur_obs, action, reward, done, old_log_prob, value)
                cur_obs = next_obs

                if self.pooling.enough_data:
                    observations, actions = self.pooling.observations, self.pooling.actions
                    rewards, dones = self.pooling.rewards, self.pooling.dones
                    values, old_log_probs = self.pooling.values, self.pooling.old_log_probs
                    self.pooling.clear_data()
                    returns = self.compute_gae(next_obs, rewards, values, dones)
                    print('start to train')
                    self._do_training(torch.FloatTensor(returns), torch.FloatTensor(values),
                                      torch.FloatTensor(old_log_probs), torch.FloatTensor(observations),
                                      torch.FloatTensor(actions))
                    torch.save(self.actor_critic.state_dict(), f'checkpoints/{self.iter_steps}.pth')
                    self.iter_steps += 1
                path_rewards += reward
                if done or self.max_steps_per_episodes == path_length:
                    break
            self.rewards_learning_prcoess.append(path_rewards)
            logger.info("Episode: %d,          Path length: %d       Reward: %f" % (episode_t + 1, path_length, path_rewards))
            # print("Episode: %d,          Path length: %d       Reward: %f" % (episode_t + 1, path_length, path_rewards))
            if len(self.reward_cnt) >= 10:
                self.reward_cnt.pop(0)
                self.reward_cnt.append(path_rewards)
            else:
                self.reward_cnt.append(path_rewards)
            if episode_t == 0:
                self.vis.line(
                    X=np.array([0]),
                    Y=np.array([100 * path_rewards - 50]),
                    win='mean rewards',
                    opts=dict(
                        xlabel='episodes',
                        ylabel='mean rewards',
                        title='mean rewards'))
                self.vis.line(
                    X=np.array([0]),
                    Y=np.array([path_length]),
                    win="path len",
                    opts=dict(
                        xlabel='episodes',
                        ylabel='path len',
                        title='path len'))
                self.vis.line(
                    X=np.array([0]),
                    Y=np.array([100 * path_rewards - 50]),
                    win="rewards",
                    opts=dict(
                        xlabel='episodes',
                        ylabel='rewards',
                        title='rewards'))
            else:
                self.vis.line(
                    X=np.array([episode_t + 1]),
                    Y=np.array([100 * (sum(self.reward_cnt) / len(self.reward_cnt) if len(self.reward_cnt) else 0) - 50]),
                    win='mean rewards',
                    update='append')
                self.vis.line(
                    X=np.array([episode_t + 1]),
                    Y=np.array([path_length]),
                    win="path len",
                    update='append')
                self.vis.line(
                    X=np.array([episode_t + 1]),
                    Y=np.array([100 * path_rewards - 50]),
                    win="rewards",
                    update='append')
            if episode_t % 3 == 0:
                eval_path_len, eval_rewards = self.eval(num_episodes=1)
                if episode_t == 0:
                    self.vis.line(X=np.array([episode_t]),
                                  Y=np.array([eval_path_len]),
                                  win='eval_path_len',
                                  opts=dict(xlabel='iter steps',
                                            ylabel='path length',
                                            title='path length'))
                    self.vis.line(X=np.array([episode_t]),
                                  Y=np.array([100 * eval_rewards - 50]),
                                  win='eval_rewards',
                                  opts=dict(xlabel='iter steps',
                                            ylabel='eval rewards',
                                            title='eval rewards'))
                else:
                    self.vis.line(X=np.array([episode_t]),
                                  Y=np.array([eval_path_len]),
                                  win='eval_path_len',
                                  update='append')
                    self.vis.line(X=np.array([episode_t]),
                                  Y=np.array([100 * eval_rewards - 50]),
                                  win='eval_rewards',
                                  update='append')

    def compute_gae(self, next_obs, rewards, values, dones, lammbda=0.95):

        _, _, value_t_1 = self.actor_critic.select_action(torch.FloatTensor(next_obs[None]))
        gae = 0
        returns = np.zeros_like(values)
        for step in reversed(range(rewards.shape[0])):
            td_delta = rewards[step] + self.gamma * (1. - dones[step]) * value_t_1 - values[step]
            value_t_1 = values[step]
            gae = self.gamma * lammbda * (1. - dones[step]) * gae + td_delta
            returns[step] = gae + values[step]
        return returns

    def _do_training(self, returns, values, old_log_probs, observations, actions):
        minibatch = max(int(self.pooling.buffer_size / self.pooling.batch_size), 1)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.pooling.buffer_size)), minibatch, True):
                new_dists, actor_entropys, new_values = self.actor_critic.compute_action(observations[index])
                new_log_probs = new_dists.log_prob(actions[index]).sum(1, keepdim=True)

                ratios = torch.exp(new_log_probs - old_log_probs[index])
                advantages = returns[index] - values[index]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1. - self.clip_epsilon, 1. + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.weight_epsilon * actor_entropys.mean()
                critic_loss = (returns[index] - new_values).pow(2).mean()
                loss = 0.5 * critic_loss + actor_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval(self, num_episodes=1):
        path_len = 0
        rewards = 0
        cur_obs = self.env.reset()
        while True:
            action = self.actor_critic.eval_action(torch.FloatTensor(cur_obs[None]))
            next_obs, reward, done, info = self.env.step(action)
            rewards += reward
            path_len += 1
            cur_obs = next_obs
            if done:
                break
        return path_len, rewards




