import numpy as np
from PIL import Image
import logging


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def sample(self):
        raise NotImplementedError

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

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
        self._image_frames = []
        self.reward_episodes = []

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("logt.txt")
        formatter = logging.Formatter()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def sample(self, render=False, save_gif=False):
        if self._current_observation is None:
            print('Episode %i: Start simulation...' % self._n_episodes)
            self._current_observation = self._normalize(self.env.reset())

        if render:
            self.env.render()
            if save_gif:
                self._image_frames.append(Image.fromarray(self.env.render(mode='rgb_array')))

        action, _ = self.policy.get_action(self._current_observation, self._n_episodes)
        # print('                        %s'%str(action))
        next_observation, reward, terminal, info = self.env.step(action)
        next_observation = self._normalize(next_observation)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            if save_gif:
                im = Image.new('RGB', self._image_frames[0].size)
                save_file = open('./gif/{}.gif'.format(self._n_episodes), 'wb')
                im.save(save_file, save_all=True, append_images=self._image_frames)

            print('            This path length is %i, total reward is %f' % (self._path_length, self._path_return))
            # if self._n_episodes % 3 == 0:
            #     self.logger.info('episode:  %d  \n', self._n_episodes)
            # self.logger.info('episode:  %d  total reward:  %.2f', self._n_episodes, self._path_return)
            self.reward_episodes.append(self._path_return)

            self._path_length = 0
            self._max_path_return = max(self._max_path_return, self._path_return)
            self._last_path_return = self._path_return
            self._path_return = 0
            self._n_episodes += 1

            self._current_observation = None
            self._image_frames = []

            self.policy.reset()
            self.env.close()
        else:
            self._current_observation = next_observation

    def batch_ready(self):
        enough_samples = self._total_samples >= self._min_pool_size
        return enough_samples

    def log_diagnostics(self):
        print('last path return =', self._last_path_return)

    def _normalize(self, obs):
        mean, std = np.array([-5.24356906e-01, -2.17277541e-04]), np.array([0.08861222, 0.00765672])
        return obs            # (obs - mean) / std



















