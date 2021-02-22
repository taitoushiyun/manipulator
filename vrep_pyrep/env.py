import time
import copy
import numpy as np
from pyrep import PyRep
from manipulator import ManipulatorPlane, Manipulator3D, ManipulatorCCPlane, ManipulatorCC3D
from dh_convert import DHModel
from os.path import dirname, join, abspath
import gym
from gym import spaces
from pyrep.objects.shape import Shape
import logging

logger = logging.getLogger('mani')
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi

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
        super(ManipulatorEnv, self).__init__()
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

        self.state_dim = self.joint_state_dim + 6  # EE_point_position, EE_point_vel, goal_position, base_position
        self.action_dim = self.joint_state_dim // 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.j_ang_idx = range(self.joint_state_dim // 2)
        self.j_vel_idx = range(self.joint_state_dim // 2, self.joint_state_dim)
        self.e_pos_idx = range(self.joint_state_dim, self.joint_state_dim + 3)
        self.e_vel_idx = range(self.joint_state_dim + 3, self.joint_state_dim + 6)
        # self.g_pos_idx = range(self.joint_state_dim + 6, self.joint_state_dim + 9)
        # self.b_pos_idx = range(self.joint_state_dim + 9, self.joint_state_dim + 12)

        self.dh_model = DHModel(self.num_joints)
        self.observation = np.zeros((self.state_dim,))
        self.last_obs = None
        self._elapsed_steps = None

        self.pr = PyRep()
        self.pr.launch(join(dirname(abspath(__file__)), env_config['scene_file']), headless=self.headless_mode)
        self.pr.start()
        self.manipulator = mani_cls(num_joints=self.num_joints,
                                    num_segments=self.num_segments,
                                    collision_cnt=self.collision_cnt)
        self.manipulator.set_control_loop_enabled(False)
        self.manipulator.set_motor_locked_at_zero_velocity(True)
        self.manipulator_ee_tip = self.manipulator.get_tip()
        self.manipulator_target = Shape("target")
        self.manipulator_base = self.manipulator.get_base()
        self.initial_config_tree = self.manipulator.get_configuration_tree()

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
        else:
            raise ValueError
        goal_theta = np.clip(theta, -3, 3)
        goal = self.dh_model.forward_kinematics(goal_theta)
        reset_state = self.dh_model.forward_kinematics(np.zeros((self.num_joints, )))
        max_rewards = np.linalg.norm(goal - reset_state, axis=-1)
        return goal_theta, goal, max_rewards

    def _get_state(self):
        state = np.zeros(self.state_dim)
        state[self.j_ang_idx] = np.asarray(self.manipulator.get_joint_positions()) * RAD2DEG
        state[self.j_vel_idx] = np.asarray(self.manipulator.get_joint_velocities()) * RAD2DEG
        state[self.e_pos_idx] = np.asarray(self.manipulator_ee_tip.get_position())
        state[self.e_vel_idx] = np.asarray(self.manipulator_ee_tip.get_velocity()[0])
        # state[self.g_pos_idx] = self.observation[self.g_pos_idx]
        # state[self.b_pos_idx] = self.observation[self.b_pos_idx]
        info = {'collision_state': self.manipulator.get_collision_result()}
        return state, info

    def normalize(self, obs):
        state = copy.deepcopy(obs)
        state[self.j_ang_idx] /= 90.
        state[self.j_vel_idx] /= 10.
        state[self.e_pos_idx[0]] = (state[self.e_pos_idx[0]] - 0.4) / .4
        state[self.e_pos_idx[2]] = (state[self.e_pos_idx[2]] - 1.) / 1.
        state[self.e_vel_idx] /= 0.5
        # state[self.g_pos_idx[0]] = (state[self.g_pos_idx[0]] - 0.4) / .4
        # state[self.g_pos_idx[2]] = (state[self.g_pos_idx[2]] - 1.) / 1.
        return state

    def reset(self):
        if self.cc_model:
            self.pr.stop()
            self.pr.start()
        else:
            self.pr.set_configuration_tree(self.initial_config_tree)
        self._elapsed_steps = 0
        self.goal_theta, self.goal, self.max_rewards = self._sample_goal()
        # self.observation[self.g_pos_idx] = np.asarray(self.goal)
        # self.observation[self.b_pos_idx] = np.asarray([0.2, 0, 1])
        self.manipulator_target.set_position(self.goal)
        self.manipulator.set_initial_joint_positions(self.initial_joint_positions)
        observation, _ = self._get_state()
        self.last_obs = observation
        return {'observation': observation.copy(),
                'achieved_goal': observation[self.e_pos_idx].copy(),
                'desired_goal': self.goal.copy()}

    def step(self, action):
        assert self._elapsed_steps is not None
        self._elapsed_steps += 1
        if self.plane_model and not self.cc_model:
            action = self.max_angles_vel * DEG2RAD * np.asarray(action)
            action = action[:, np.newaxis]
            action = np.concatenate((np.zeros((self.action_dim, 1)), action), axis=-1).flatten()
        elif self.plane_model and self.cc_model:
            action = self.max_angles_vel * DEG2RAD * np.asarray(action)
            action = action[:, np.newaxis]
            action = np.concatenate((np.zeros((self.num_segments, 1)), action), axis=-1).flatten()
        else:
            action = self.max_angles_vel * DEG2RAD * np.asarray(action)
        self.manipulator.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        observation, info = self._get_state()
        achieved_goal = observation[self.e_pos_idx]
        reward = self.compute_reward(achieved_goal, self.goal, None)
        done = np.linalg.norm(achieved_goal - self.goal, axis=-1) <= self.distance_threshold
        info['is_success'] = done
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        if any(info['collision_state']):
            done = True
        self.last_obs = observation
        obs_dict = {'observation': observation.copy(),
                    'achieved_goal': observation[self.e_pos_idx].copy(),
                    'desired_goal': self.goal.copy()}
        return obs_dict, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal - goal, axis=-1)

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense distance':
            return -d
        elif self.reward_type == 'dense potential':
            last_achieved_goal = self.last_obs[self.e_pos_idx]
            d_last = np.linalg.norm(last_achieved_goal - self.goal, axis=-1)
            return -d + d_last
        else:
            raise ValueError('reward type wrong')

    def end_simulation(self):
        self.pr.stop()
        self.pr.shutdown()


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
        'goal_set': 'hard',
        'max_episode_steps': 100,
        'collision_cnt': 27,
        'scene_file': 'simple_24_1.ttt',
        'headless_mode': False,
    }
    env = ManipulatorEnv(env_config)
    print('env created success')
    # action_ = [-1, 1]

    # for i in range(100):
    #     action_ = np.random.uniform(low=-1, high=1, size=(2, ))
    #     step = 0
    #     obs = env.reset()
    #     while True:
    #         obs, reward, done, info = env.step(action_)
    #         step += 1
    #         if done:
    #             step = 0
    #             break
    obs = env.reset()
    print(obs[env.e_pos_idx])
    time.sleep(1)
    env.end_simulation()
