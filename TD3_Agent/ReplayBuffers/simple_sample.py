import numpy as np
import logging
from _collections import deque

class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env, self.policy, self.pooling = None, None, None

    def initialize(self, env, policy, pool):
        self.env, self.policy, self.pooling = env, policy, pool

    def sample(self):
        raise NotImplementedError

    def random_batch(self):
        return self.pooling.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)
        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 1
        self._current_observation = None
        self._total_samples = 0
        self.reward_episodes = []
        import visdom
        self.vis = visdom.Visdom(port=6016, env='td3_2')
        self.vis.line(X=[0], Y=[0], win='reward', opts=dict(Xlabel='episode', Ylabel='reward', title='reward'))
        self.vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
        self.vis.line(X=[0], Y=[0], win='mean reward',
                 opts=dict(Xlabel='episode', Ylabel='mean reward', title='mean reward'))
        self.vis.line(X=[0], Y=[0], win='eval reward', opts=dict(Xlabel='episode', Ylabel='reward', title='eval reward'))
        self.vis.line(X=[0], Y=[0], win='eval path len', opts=dict(Xlabel='episode', Ylabel='len', title='eval path len'))
        self.queue = deque(maxlen=10)

    def sample(self, render=False):
        if self._current_observation is None:
            print('Episode %i: Start simulation...' % self._n_episodes)
            self._current_observation = self.env.reset()

        action, _ = self.policy.get_action(self._current_observation, self._n_episodes)
        next_obs, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pooling.add_sample(self._current_observation,
                                action, reward, terminal, next_obs)

        if terminal or self._path_length >= self._max_path_length:
            self.env.close()
            self.queue.append(self._path_return)
            self.vis.line(X=[self._n_episodes], Y=[self._path_return], win='reward', update='append')
            self.vis.line(X=[self._n_episodes], Y=[self._path_length], win='path len', update='append')
            self.vis.line(X=[self._n_episodes], Y=[sum(self.queue) / len(self.queue)], win='mean reward', update='append')
            print('            This path length is %i, total reward is %f' % (self._path_length, self._path_return))
            self.reward_episodes.append(self._path_return)
            self._path_length = 0
            self._max_path_return = max(self._max_path_return, self._path_return)
            self._last_path_return = self._path_return
            self._path_return = 0
            self._n_episodes += 1

            self._current_observation = None
            self.policy.reset()

            total_reward = 0
            total_len = 0
            eval_obs = self.env.reset()
            while True:
                eval_action, _ = self.policy.eval_action(eval_obs, self._n_episodes)
                eval_next_obs, eval_reward, eval_done, eval_info = self.env.step(eval_action)
                total_reward += eval_reward
                total_len += 1
                eval_obs = eval_next_obs
                if eval_done or total_len >= self._max_path_length:
                    break
            self.vis.line(X=[self._n_episodes], Y=[total_reward], win='eval reward', update='append')
            self.vis.line(X=[self._n_episodes], Y=[total_len], win='eval path len', update='append')


        else:
            self._current_observation = next_obs

    def batch_ready(self):
        enough_samples = self._total_samples >= self._min_pool_size
        return enough_samples

    def log_diagnostics(self):
        print('last path return =', self._last_path_return)

    @property
    def num_current_total_steps(self):
        return self._total_samples














