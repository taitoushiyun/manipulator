import numpy as np
import time
import gym
from gym import spaces

GOAL_DICT = {}

class EnvTest(gym.Env):
    def __init__(self, env_config):
        super(EnvTest, self).__init__()
        self.pos = np.array([1.4, 0, 1])
        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.action_space = spaces.Box(low=-1., high=1, shape=(3,), dtype=np.float32)
        self._elapsed_steps = 0
        self.max_angles_vel = 0.05
        self._max_episode_steps = 100

    def reset(self, goal_set=None, i_epoch=0):
        self._elapsed_steps = 0
        self.goal = np.array([0.8, 0, 0.73])
        self.pos = np.array([1.4, 0, 1])
        return {'observation': self.pos.copy(),
                'desired_goal': self.goal.copy(),
                'achieved_goal': self.pos.copy()}

    def step(self, action):
        self._elapsed_steps += 1
        assert action.size == 3
        next_pos = self.pos + self.max_angles_vel * action
        x_y_limit = 0.03 <= np.linalg.norm(next_pos[:2] - self.goal[:2]) <= 0.1
        z_limit = 0.7 <= next_pos[2] <= 0.8
        if x_y_limit and z_limit:
            if x_y_limit:
                next_pos[:2] = self.pos[:2]
            if z_limit:
                if action[2] > 0:
                    next_pos[2] = 0.7
                elif action[2] < 0:
                    next_pos[2] = 0.8
                else:
                    next_pos[2] = self.pos[2]

        self.pos = next_pos
        reward = -(np.linalg.norm(self.goal - self.pos, axis=-1) > 0.02).astype(np.float32)
        done = np.linalg.norm(self.goal - self.pos, axis=-1) < 0.02
        info = {'is_success': done}
        return {'observation': self.pos.copy(),
                'desired_goal': self.goal.copy(),
                'achieved_goal': self.pos.copy()}, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        return -(d > 0.02).astype(np.float32)


if __name__ == '__main__':
    env_config = {'max_angles_vel': 0.05}
    env = EnvTest(env_config)
    time_a = time.time()
    for i in range(100):
        env.reset()
        for j in range(100):
            action = np.random.randn(3).clip(-1, 1)
            obs, reward, done, info = env.step(action)
    time_b = time.time()
    print(time_b -time_a)

