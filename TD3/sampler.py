import numpy as np
from _collections import deque

class ReplayBuffer(object):
    def __init__(self, max_memory_size, obs_dims, act_dims):
        self.max_memory_size = max_memory_size
        self._top = 0
        self.observations = np.zeros((max_memory_size, obs_dims))
        self.actions = np.zeros((max_memory_size, act_dims))
        self.next_observations = np.zeros((max_memory_size, obs_dims))
        self.rewards = np.zeros((max_memory_size, 1))
        self.terminals = np.zeros((max_memory_size, 1))
        self._size = 0
        self.flag = True

    def add_sample(self, obs, act, reward, next_obs, terminal):
        if terminal:
            terminal = 1
        else:
            terminal = 0
        self.observations[self._top] = obs
        self.actions[self._top] = act
        self.rewards[self._top] = reward
        self.next_observations[self._top] = next_obs
        self.terminals[self._top] = terminal
        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self.max_memory_size
        if self._size < self.max_memory_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            terminals=self.terminals[indices],
            next_observations=self.next_observations[indices],
        )

    @property
    def total_size(self):
        return self._size


class Sampler(object):
    def __init__(self, env, policy, pool, min_pool_size, max_episode_steps, code_version):
        self.env = env
        self.policy = policy
        self.pool = pool
        self.min_pool_size = min_pool_size
        self.path_len = 0
        self.path_reward = 0
        self.n_episodes = 0
        self.total_samples = 0
        self.cur_obs = None
        self.max_episode_steps = max_episode_steps
        import visdom
        self.vis = visdom.Visdom(port=6016, env=code_version)
        self.vis.line(X=[0], Y=[0], win='reward', opts=dict(Xlabel='episode', Ylabel='reward', title='reward'))
        self.vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
        self.vis.line(X=[0], Y=[0], win='mean reward',
                      opts=dict(Xlabel='episode', Ylabel='mean reward', title='mean reward'))
        self.vis.line(X=[0], Y=[0], win='eval reward',
                      opts=dict(Xlabel='episode', Ylabel='reward', title='eval reward'))
        self.vis.line(X=[0], Y=[0], win='eval path len',
                      opts=dict(Xlabel='episode', Ylabel='len', title='eval path len'))
        self.queue = deque(maxlen=10)

    def sample(self):
        if self.cur_obs is None:
            self.cur_obs = self.env.reset()
        action = self.policy.select_action(self.cur_obs)
        next_obs, reward, done, info = self.env.step(action)
        self.path_len += 1
        self.path_reward += reward
        self.total_samples += 1
        self.pool.add_sample(self.cur_obs[:], action, reward, next_obs, done)
        self.cur_obs = next_obs
        if done or self.path_len >= self.max_episode_steps:
            print(f'episode {self.n_episodes}, reward is {self.path_reward}, len is {self.path_len}')
            self.queue.append(self.path_reward)
            self.vis.line(X=[self.n_episodes], Y=[self.path_reward], win='reward', update='append')
            self.vis.line(X=[self.n_episodes], Y=[self.path_len], win='path len', update='append')
            self.vis.line(X=[self.n_episodes], Y=[sum(self.queue) / len(self.queue)], win='mean reward',
                          update='append')
            self.cur_obs = None
            self.path_len = 0
            self.path_reward = 0
            self.n_episodes += 1

            total_reward = 0
            total_len = 0
            eval_obs = self.env.reset()
            while True:
                eval_action = self.policy.eval_action(eval_obs)
                eval_next_obs, eval_reward, eval_done, eval_info = self.env.step(eval_action)
                total_reward += eval_reward
                total_len += 1
                eval_obs = eval_next_obs
                if eval_done or total_len >= self.max_episode_steps:
                    break
            self.vis.line(X=[self.n_episodes], Y=[total_reward], win='eval reward', update='append')
            self.vis.line(X=[self.n_episodes], Y=[total_len], win='eval path len', update='append')

    def batch_ready(self):
        return self.total_samples >= self.min_pool_size











