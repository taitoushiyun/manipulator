import importmagic
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import visdom
import logging
from collections import deque
import os
logger = logging.getLogger('mani')


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
        self.model_dir = os.path.join('checkpoints', args.code_version)
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self):
        vis = visdom.Visdom(port=self.args.visdom_port, env=self.args.code_version)
        vis.line(X=[0], Y=[0], win='result', opts=dict(Xlabel='episode', Ylabel='result', title='result'))
        vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
        vis.line(X=[0], Y=[0], win='success rate',
                 opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='success rate'))
        vis.line(X=[0], Y=[0], win='eval result',
                 opts=dict(Xlabel='episode', Ylabel='eval result', title='eval result'))
        vis.line(X=[0], Y=[0], win='eval path len', opts=dict(Xlabel='episode', Ylabel='len', title='eval path len'))
        vis.line(X=[0], Y=[0], win='eval success rate',
                 opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='eval success rate'))
        if self.args.goal_set != 'random':
            vis.line(X=[0], Y=[0], win='reward', opts=dict(Xlabel='episode', Ylabel='reward', title='reward'))
            vis.line(X=[0], Y=[0], win='mean reward', opts=dict(Xlabel='episode', Ylabel='reward', title='mean reward'))
            vis.line(X=[0], Y=[0], win='eval reward', opts=dict(Xlabel='episode', Ylabel='reward', title='eval reward'))
        result_deque = deque(maxlen=20)
        score_deque = deque(maxlen=10)
        eval_result_queue = deque(maxlen=10)

        for i_episode in range(self.num_episodes):
            cur_obs = self.env.reset()
            path_length, path_rewards = 0, 0.
            while True:
                path_length += 1
                action, old_log_prob, value = self.actor_critic.select_action(torch.FloatTensor(cur_obs[None]))
                next_obs, reward, done, info = self.env.step(action)
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
                    torch.save(self.actor_critic.state_dict(), os.path.join(self.model_dir, f'{i_episode}.pth'))
                    self.iter_steps += 1
                path_rewards += reward
                if done or self.max_steps_per_episodes == path_length:
                    result = 0.
                    if done and path_length < self.args.max_episode_steps and not any(info['collision_state']):
                        result = 1.
                    break
            result_deque.append(result)
            score_deque.append(path_rewards)
            success_rate = np.mean(result_deque)
            mean_score = np.mean(score_deque)
            logger.info(
                "Episode: %d,          Path length: %d       result: %f       reward: %f"
                % (i_episode, path_length, result, path_rewards))
            if self.args.goal_set != 'random':
                if self.args.reward_type == 'dense potential':
                    vis.line(X=[i_episode], Y=[(path_rewards - self.env.max_rewards) * 100], win='reward', update='append')
                    vis.line(X=[i_episode], Y=[(mean_score - self.env.max_rewards) * 100], win='mean reward',
                             update='append')
                if self.args.reward_type == 'dense distance':
                    vis.line(X=[i_episode], Y=[path_rewards], win='reward', update='append')
                    vis.line(X=[i_episode], Y=[mean_score], win='mean reward', update='append')
            vis.line(X=[i_episode], Y=[result], win='result', update='append')
            vis.line(X=[i_episode], Y=[path_length], win='path len', update='append')
            vis.line(X=[i_episode], Y=[success_rate * 100], win='success rate', update='append')
            if i_episode % self.args.eval_freq == 0:
                eval_path_len, eval_rewards, eval_result = self.eval(num_episodes=self.args.eval_times)
                eval_result_queue.append(eval_result)
                eval_success_rate = np.mean(eval_result_queue)
                logger.info(
                    "Eval Episode: %d,          Path length: %d       result: %f" % (i_episode, eval_path_len, eval_result))
                vis.line(X=[i_episode], Y=[eval_result], win='eval result', update='append')
                vis.line(X=[i_episode], Y=[eval_path_len], win='eval path len', update='append')
                vis.line(X=[i_episode], Y=[eval_success_rate * 100], win='eval success rate', update='append')
                if self.args.goal_set != 'random':
                    if self.args.reward_type == 'dense potential':
                        vis.line(X=[i_episode], Y=[100 * (eval_rewards - self.env.max_rewards)], win='eval reward',
                                 update='append')
                    if self.args.reward_type == 'dense distance':
                        vis.line(X=[i_episode], Y=[eval_rewards], win='eval reward', update='append')

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
        result = 0
        for i in range(num_episodes):
            cur_obs = self.env.reset()
            while True:
                action = self.actor_critic.eval_action(torch.FloatTensor(cur_obs[None]))
                next_obs, reward, done, info = self.env.step(action)
                rewards += reward
                path_len += 1
                cur_obs = next_obs
                if done:
                    eval_result = 0.
                    if done and path_len < self.args.max_episode_steps and not any(info['collision_state']):
                        eval_result = 1.
                    result += eval_result
                    break
        return path_len / num_episodes, rewards / num_episodes, result / num_episodes




