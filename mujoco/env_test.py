import numpy as np
import time
import gym
from gym import spaces
import os
import mujoco_py

DEFAULT_SIZE = 500
GOAL_DICT = {}

class EnvTest(gym.Env):
    def __init__(self, env_config):
        super(EnvTest, self).__init__()
        self.plane_model = env_config['plane_model']
        self.pos = np.array([1.4, 0, 1])
        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        if self.plane_model:
            self.action_dim = 2
        else:
            self.action_dim = 3
        self.action_space = spaces.Box(low=-1., high=1, shape=(self.action_dim,), dtype=np.float32)
        self._elapsed_steps = 0
        self.max_angles_vel = 0.05
        self._max_episode_steps = 50

        model_xml_path = os.path.join(os.path.dirname(__file__), 'mani', env_config['scene_file'])
        if not os.path.exists(model_xml_path):
            raise IOError('File {} does not exist'.format(model_xml_path))
        model = mujoco_py.load_model_from_path(model_xml_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps=env_config['n_substeps'])
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

    def reset(self, eval=False, i_epoch=0):
        self._elapsed_steps = 0
        self.goal = np.array([0.8, 0, 0.73])
        self.pos = np.array([1.4, 0, 1])
        if self.plane_model:
            self.goal = np.array([self.goal[0], self.goal[2]])
            self.pos = np.array([self.pos[0], self.pos[2]])
        return {'observation': self.pos.copy(),
                'desired_goal': self.goal.copy(),
                'achieved_goal': self.pos.copy()}

    def step(self, action):
        self._elapsed_steps += 1
        assert action.size == self.action_dim
        if self.plane_model:
            next_pos = self.pos + self.max_angles_vel * action
            x_limit = 0.03 <= np.absolute(next_pos[0] - self.goal[0]) <= 0.1
            z_limit = 0.7 <= next_pos[1] <= 0.8
            if x_limit and z_limit:
                if x_limit:
                    next_pos[0] = self.pos[0]
                if z_limit:
                    if action[1] > 0:
                        next_pos[1] = 0.7
                    elif action[1] < 0:
                        next_pos[1] = 0.8
                    else:
                        next_pos[1] = self.pos[1]
        else:
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

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _env_setup(self):
        pass

    def _viewer_setup(self):
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        site_tip_id = self.sim.model.site_name2id('robot0:tip')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.model.site_pos[site_tip_id] = self.pos - sites_offset[1]
        self.sim.forward()


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

