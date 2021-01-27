import time
import numpy as np
from pyrep import PyRep
from manipulator import ManipulatorPlane, Manipulator3D, ManipulatorCCPlane, ManipulatorCC3D
from dh_convert import DHModel
from os.path import dirname, join, abspath
import gym
from gym import spaces
from pyrep.objects.shape import Shape

DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi

GOAL = {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20],
        'hard': [0, 20, 0, 15, 0, 20, 0, 20, 0, 20],
        'super hard': [0, -50, 0, -50, 0, -50, 0, -20, 0, -10]}


class ManipulatorEnv(gym.Env):
    def __init__(self, env_config):
        super(ManipulatorEnv, self).__init__()
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

        self.state_dim = self.joint_state_dim + 12  # EE_point_position, EE_point_vel, goal_position, base_position
        self.action_dim = self.joint_state_dim // 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.j_ang_idx = range(self.joint_state_dim // 2)
        self.j_vel_idx = range(self.joint_state_dim // 2, self.joint_state_dim)
        self.e_pos_idx = range(self.joint_state_dim, self.joint_state_dim + 3)
        self.e_vel_idx = range(self.joint_state_dim + 3, self.joint_state_dim + 6)
        self.g_pos_idx = range(self.joint_state_dim + 6, self.joint_state_dim + 9)
        self.b_pos_idx = range(self.joint_state_dim + 9, self.joint_state_dim + 12)

        self.dh_model = DHModel(self.num_joints)
        self.observation = np.zeros((self.state_dim,))
        self.last_obs = None
        self._elapsed_steps = None

        self.pr = PyRep()
        self.pr.launch(join(dirname(abspath(__file__)), env_config['scene_file']), headless=self.headless_mode)
        self.pr.start()
        self.agent = mani_cls(num_joints=self.num_joints,
                              num_segments=self.num_segments,
                              collision_cnt=self.collision_cnt)
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.agent_target = Shape("target")
        self.agent_base = self.agent.get_base()
        self.initial_config_tree = self.agent.get_configuration_tree()

    def _sample_goal(self):
        if self.goal_set in ['easy', 'hard', 'super hard']:
            theta = np.asarray(GOAL[self.goal_set]) * DEG2RAD
        elif self.goal_set == 'random':
            if self.plane_model:
                theta = np.vstack((np.zeros((self.action_dim,)),
                                   45 * DEG2RAD * np.random.uniform(low=-1, high=1,
                                                                    size=(self.action_dim,)))).T.flatten()
            else:
                theta = 45 * DEG2RAD * np.random.uniform(low=-1, high=1, size=(self.action_dim, ))
        else:
            raise ValueError
        goal_theta = np.clip(theta, -3, 3)
        # print(f'goal sample for joints is {goal_theta}')
        goal = self.dh_model.forward_kinematics(goal_theta)
        reset_state = self.dh_model.forward_kinematics(np.zeros((self.num_joints, )))
        max_rewards = np.linalg.norm(goal - reset_state, axis=-1)
        # print(f'goal sample for end point is {goal}')
        return goal_theta, goal, max_rewards

    def _get_state(self):
        state = np.zeros(self.state_dim)
        state[self.j_ang_idx] = np.asarray(self.agent.get_joint_positions()) * RAD2DEG
        state[self.j_vel_idx] = np.asarray(self.agent.get_joint_velocities()) * RAD2DEG
        state[self.e_pos_idx] = np.asarray(self.agent_ee_tip.get_position())
        state[self.e_vel_idx] = np.asarray(self.agent_ee_tip.get_velocity()[0])
        state[self.g_pos_idx] = self.observation[self.g_pos_idx]
        state[self.b_pos_idx] = self.observation[self.b_pos_idx]
        # observation = np.concatenate([self.agent.get_joint_positions(),
        #                               self.agent.get_joint_velocities(),
        #                               self.agent_ee_tip.get_position(),
        #                               self.agent_ee_tip.get_velocity(),
        #                               self.agent_target.get_position(),
        #                               self.agent_base.get_position()])
        info = {'collision_state': self.agent.get_collision_result()}
        return state, info

    def reset(self):
        self.pr.set_configuration_tree(self.initial_config_tree)
        self._elapsed_steps = 0
        self.goal_theta, self.goal, self.max_rewards = self._sample_goal()
        self.observation[self.g_pos_idx] = np.asarray(self.goal)
        self.observation[self.b_pos_idx] = np.asarray([0.2, 0, 1])
        self.agent_target.set_position(self.goal)
        self.agent.set_initial_joint_positions(self.initial_joint_positions)
        observation, _ = self._get_state()
        self.last_obs = observation
        return observation

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
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        observation, info = self._get_state()
        reward = self.cal_reward(observation[self.e_pos_idx], self.goal)
        done = np.linalg.norm(observation[self.e_pos_idx] - self.goal, axis=-1) <= self.distance_threshold
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        if any(info['collision_state']):
            done = True
        self.last_obs = observation
        return observation, reward, done, info

    def cal_reward(self, achieved_goal, goal):

        def dense_reward(d):
            return -d
        d = np.linalg.norm(achieved_goal - goal, axis=-1)

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            # return -d
            d_last = np.linalg.norm(self.last_obs[self.e_pos_idx] - self.goal, axis=-1)
            # print(f'reward now is {dense_reward(d)}, reward last is {dense_reward(d_last)}')
            return dense_reward(d) - dense_reward(d_last)

    def end_simulation(self):
        self.pr.stop()
        self.pr.shutdown()

if __name__ == '__main__':
    goal_index = {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20],
                  'hard': [0, 20, 0, 15, 0, 20, 0, 20, 0, 20],
                  'super hard': [0, -50, 0, -50, 0, -50, 0, -20, 0, -10]}
    env_config = {
        'distance_threshold': 0.02,
        'reward_type': 'dense',
        'max_angles_vel': 10,  # 10degree/s
        'num_joints': 12,
        'num_segments': 2,
        'cc_model': False,
        'plane_model': True,
        'goal_set': 'random',
        'max_episode_steps': 100,
        'collision_cnt': 15,
        'scene_file': 'simple_12_1.ttt',
        'headless_mode': False,
    }
    env = ManipulatorEnv(env_config)
    print('env created success')
    action_ = [1, 1, 1, -1, -1, -1]
    # action_ = np.random.uniform(-1, 1, size=(6,))

    lines = []
    for i in range(1):
        step = 0
        line = []
        obs = env.reset()
        # print(obs[:1])
        while True:
            # time_a = time.time()
            obs, reward, done, info = env.step(action_)
            # time_b = time.time()
            # print(time_b - time_a)
            # print(obs[0])
            line.append(obs[0])
            step += 1
            if done:
                lines.append(line)
                step = 0
                break
    # from matplotlib import pyplot as plt
    # for i in range(1):
    #     plt.plot(lines[i])
    # plt.show()

    env.end_simulation()