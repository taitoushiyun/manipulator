import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_memory_size, obs_dims, act_dims):
        self.max_memory_size = max_memory_size
        self._top = 0
        self.observations = np.zeros((max_memory_size, obs_dims), np.float)
        self.actions = np.zeros((max_memory_size, act_dims))
        self.next_observations = np.zeros((max_memory_size, obs_dims))
        self.rewards = np.zeros(max_memory_size)
        self.terminals = np.zeros(max_memory_size)
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
    def __init__(self, env, policy, pool, min_pool_size, max_episode_steps):
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

    def sample(self):
        if self.cur_obs is None:
            self.cur_obs = self.env.reset()
        action = self.policy.select_action(self.cur_obs)
        next_obs, reward, done, info = self.env.step(action)
        self.path_len += 1
        self.path_reward += reward
        self.pool.add_sample(self.cur_obs[:], action, reward, next_obs, done)
        self.cur_obs = next_obs
        if done or self.path_len >= self.max_episode_steps:
            self.cur_obs = self.env.reset()
            self.path_len = 0
            self.path_reward = 0
            self.n_episodes += 1

    def batch_ready(self):
        return self.total_samples >= self.min_pool_size











