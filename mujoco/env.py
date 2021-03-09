import time
import copy
import numpy as np
from dh_convert import DHModel
from manipulator import ManipulatorPlane, Manipulator3D, ManipulatorCCPlane, ManipulatorCC3D
import os
import gym
from gym import spaces
from gym.utils import seeding
from gym import utils
import mujoco_py
import logging


logger = logging.getLogger('mani')
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi
DEFAULT_SIZE = 500

# (cc_model, plane_model)
GOAL = {(True, True): {'easy': [0, 20, 0, 20, 0, 20, 0, -10, 0, -10, 0, -10],
                       'hard': [0, 20, 0, 20, 0, 20, 0,  20, 0,  20, 0,  20],
                       'super hard': [0, -45, 0, -45, 0, -45, 0, -30, 0, -30, 0, -30]},
                       # 'super hard': [0, -50, 0, -50, 0, -20, 0, 40, 0, 30, 0, 0]},
        (True, False): {'easy': [20, 20, 20, 20, 20, 20, -10, -10, -10, -10, -10, -10],
                        'hard': [20, 20, 20, 20, 20, 20, 20,  20, 20,  20, 20,  20],
                        'super hard': [-45, -45, -45, -45, -45, -45, -30, -30, -30, -30, -30, -30]},
        (False, True): {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20, 0,  0],
                        'hard': [0, 20, 0, 15, 0,  20, 0,  20, 0, 20, 0, -10],
                        'super hard': [0, -50, 0, -50, 0, -20, 0, 40, 0, 30, 0, 0]},
        (False, False): {'easy': [20, 20, 20, 20, -10, -10, -15, -15, 20, 20, 0, 0],
                         'hard': [20, 20, 15, 15, 20, 20, 20, 20, 20, 20, -10, -10],
                         'super hard': [-50, -50, -50, -50, -20, -20, 40, 40, 30, 30, 0, 0]}}


class ManipulatorEnv(gym.Env):
    def __init__(self, env_config):
        logger.info(GOAL)
        self.max_angles_vel = env_config['max_angles_vel']
        self.distance_threshold = env_config['distance_threshold']
        self.reward_type = env_config['reward_type']
        self.num_joints = env_config['num_joints']
        self.num_segments = env_config['num_segments']
        self.cc_model = env_config['cc_model']
        self.plane_model = env_config['plane_model']
        self.goal_set = env_config['goal_set']
        self._max_episode_steps = env_config['max_episode_steps']
        self.collision_cnt = env_config['collision_cnt']
        self.headless_mode = env_config['headless_mode']
        self.random_initial_state = env_config.get('random_initial_state', False)

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

        self.seed()
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())

        if self.cc_model:
            assert '_cc' in env_config['scene_file'], "scene file not match env configuration"

        if self.plane_model and self.cc_model:
            self.joint_state_dim = 2 * self.num_segments
            mani_cls = ManipulatorCCPlane
            self.initial_joint_positions = self.initial_joint_velocities = np.zeros((2 * self.num_segments,))
        elif self.plane_model and not self.cc_model:
            self.joint_state_dim = self.num_joints
            mani_cls = ManipulatorPlane
            self.initial_joint_positions = self.initial_joint_velocities = np.zeros((self.num_joints,))
        elif not self.plane_model and self.cc_model:
            self.joint_state_dim = 4 * self.num_segments
            mani_cls = ManipulatorCC3D
            self.initial_joint_positions = self.initial_joint_velocities = np.zeros((2 * self.num_segments,))
        elif not self.plane_model and not self.cc_model:
            self.joint_state_dim = 2 * self.num_joints
            mani_cls = Manipulator3D
            self.initial_joint_positions = self.initial_joint_velocities = np.zeros((self.num_joints,))
        else:
            raise ValueError
        self.manipulator = mani_cls(sim=self.sim,
                                    num_joints=self.num_joints,
                                    num_segments=self.num_segments,
                                    collision_cnt=self.collision_cnt)
        self.state_dim = self.joint_state_dim + 6  # EE_point_position, EE_point_vel, goal_position, base_position
        self.action_dim = self.joint_state_dim // 2

        self.dh_model = DHModel(self.num_joints)
        self.goal = None
        self.goal_index = -1
        self.has_reset = False
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.action_space = spaces.Box(low=-1., high=1, shape=(self.action_dim,), dtype=np.float32)

        self.j_ang_idx = range(self.joint_state_dim // 2)
        self.j_vel_idx = range(self.joint_state_dim // 2, self.joint_state_dim)
        self.e_pos_idx = range(self.joint_state_dim, self.joint_state_dim + 3)
        self.e_vel_idx = range(self.joint_state_dim + 3, self.joint_state_dim + 6)
        # self.g_pos_idx = range(self.joint_state_dim + 6, self.joint_state_dim + 9)

        self.last_obs = None
        self._elapsed_steps = None

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self._elapsed_steps is not None
        self._elapsed_steps += 1
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        done = np.linalg.norm(obs['achieved_goal'] - self.goal, axis=-1) <= self.distance_threshold
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        self.last_obs = obs
        return obs, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        self.goal_theta, self.goal, self.max_rewards = self._sample_goal()
        self._reset_sim()
        obs = self._get_obs()
        self.last_obs = obs
        return obs

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

    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal - goal, axis=-1)

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense distance':
            return -d
        elif self.reward_type == 'dense potential':
            last_achieved_goal = self.last_obs['achieved_goal']
            d_last = np.linalg.norm(last_achieved_goal - self.goal, axis=-1)
            return -d + d_last
        else:
            raise ValueError('reward type wrong')

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

    def _get_obs(self):
        joint_pos = self.manipulator.get_joint_positions()
        joint_vel = self.manipulator.get_joint_velocities()
        end_pos = self.sim.data.get_site_xpos('robot0:tip')
        end_vel = self.sim.data.get_site_xvelp('robot0:tip')
        achieved_goal = end_pos
        obs = np.concatenate([joint_pos, joint_vel, end_pos, end_vel])
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy() if self.goal is not None else self.goal,
        }

    def _set_action(self, action):
        if self.plane_model and not self.cc_model:
            action = self.max_angles_vel * np.asarray(action)
            action = action[:, np.newaxis]
            action = np.concatenate((np.zeros((self.action_dim, 1)), action), axis=-1).flatten()
        elif self.plane_model and self.cc_model:
            action = self.max_angles_vel * np.asarray(action)
            action = action[:, np.newaxis]
            action = np.concatenate((np.zeros((self.num_segments, 1)), action), axis=-1).flatten()
        else:
            action = self.max_angles_vel * np.asarray(action)
        self.manipulator.set_joint_target_velocities(action)

    def _reset_sim(self):
        if self.random_initial_state:
            if self.has_reset:
                return
            else:
                self.has_reset = True
        #     # if self.goal is not None:
        #     #     initial_state_joints, _, _ = self._sample_goal()
        #     #     # TODO set initial position
        #     #     initial_state = initial_state_joints * RAD2DEG
        #     #     assert len(self.initial_state.qpos) == len(initial_state)
        #     #     for i in range(len(initial_state)):
        #     #         self.initial_state.qpos[i] = initial_state[i]
        self.sim.set_state(self.initial_state)
        self.sim.forward()

    def _sample_goal(self):
        if self.goal_set in ['easy', 'hard', 'super hard']:
            theta = np.asarray(GOAL[(self.cc_model, self.plane_model)][self.goal_set]) * DEG2RAD
        elif self.goal_set == 'random':
            if self.plane_model and not self.cc_model:
                theta = np.vstack((np.zeros((self.action_dim,)),
                                   45 * DEG2RAD * np.random.uniform(low=-1, high=1,
                                                                    size=(self.action_dim,)))).T.flatten()
            elif not self.plane_model and not self.cc_model:
                theta = 45 * DEG2RAD * np.random.uniform(low=-1, high=1, size=(self.action_dim, ))
            elif self.plane_model and self.cc_model:
                theta = 45 * DEG2RAD * np.random.uniform(-1, 1, size=(self.action_dim, 1)) \
                        * np.ones((self.action_dim, self.num_joints // (2 * self.action_dim)))
                theta = theta.flatten()
                theta = np.vstack((np.zeros((self.num_joints // 2,)),
                                   theta)).T.flatten()
            elif not self.plane_model and self.cc_model:
                theta = 45 * DEG2RAD * np.random.uniform(-1, 1, size=(self.action_dim, 1)) \
                        * np.ones((self.action_dim, self.num_joints // self.action_dim))
                theta = theta.flatten()
            else:
                raise ValueError
        elif isinstance(self.goal_set, str) and self.goal_set.startswith('block'):
            if self.goal_set == 'block0':
                return None, np.array([0.7, 0, 0.8]), 0
            elif self.goal_set == 'block1':
                return None, np.array([0.6, 0, 1.2]), 0
            elif self.goal_set == 'block2':
                return None, np.array([0.5, 0, 0.73]), 0
            elif self.goal_set == 'block3':
                return None, np.array([0.8, 0, 0.73]), 0
            elif self.goal_set == 'block4':
                return None, np.array([1.2, 0, 0.8]), 0
        elif isinstance(self.goal_set, str) and self.goal_set.startswith('draw'):
            if self.goal_set == 'draw0':
                self.goal_index += 1
                goal = np.array([1.2 + 0.2 * np.cos(self.goal_index * 2 * np.pi / 18), 0.2 * np.sin(self.goal_index * 2 * np.pi / 18), 1])
                # print(goal)
                return None, goal, 0
        else:
            raise ValueError
        goal_theta = np.clip(theta, -3, 3)
        goal = self.dh_model.forward_kinematics(goal_theta)
        reset_state = self.dh_model.forward_kinematics(np.zeros((self.num_joints, )))
        max_rewards = np.linalg.norm(goal - reset_state, axis=-1)
        return goal_theta, goal, max_rewards

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self):
        pass

    def _viewer_setup(self):
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _step_callback(self):
        pass


if __name__ == '__main__':
    goal_index = {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20],
                  'hard': [20, 20, 15, 15, 20, 20, 20, 20, 20, 20, -10, -10, 20, 20, 15, 15, 20, 20, 20, 20, 20, 20, -10, -10],
                  'super hard': [0, -50, 0, -50, 0, -50, 0, -20, 0, -10]}
    env_config = {
        'distance_threshold': 0.02,
        'reward_type': 'dense distance',
        'max_angles_vel': 10,  # 10degree/s
        'num_joints': 24,
        'num_segments': 2,
        'cc_model': False,
        'plane_model': False,
        'goal_set': 'random',
        'max_episode_steps': 20,
        'collision_cnt': 15,
        'scene_file': 'mani_env_24.xml',
        'headless_mode': False,
        'n_substeps': 100,
        'random_initial_state': False,
    }
    env = ManipulatorEnv(env_config)
    # action_ = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # obs = env.reset()
    # while True:
    #     pass

    time_a = time.time()
    lines = []
    for i in range(1):
        line = []
        obs = env.reset()
        print(obs['achieved_goal'])
        # env.render()
        while True:
            pass
        # last_obs = obs['observation'][1] * RAD2DEG
        # for j in range(env_config['max_episode_steps']):
        #     time_a = time.time()
        #     env.render()
        #     if j<10:
        #         # action_ = np.array([-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0])
        #         # action_ = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) * 2
        #         # # action_ = np.ones((6, )) * j
        #         action_ = np.ones((24, )) * -1
        #     else:
        #         action_ = np.zeros((env.action_dim, ))
        #     obs, reward, done, info = env.step(action_)
        #     # print(obs['observation'][1] * RAD2DEG - last_obs)
        #     # last_obs = obs['observation'][1] * RAD2DEG
        #     time_b = time.time()
        #     print(time_b - time_a)
        #     # print(env.sim.model.opt.timestep)
        #
        #     # print(obs['observation'].shape)
        #     line.append(obs['observation'][1] * RAD2DEG)
        # lines.append(line)


    # time_b = time.time()
    # print(time_b - time_a)

    # from matplotlib import pyplot as plt
    # for i in range(1):
    #     plt.plot(lines[0])
    # plt.show()


