import time
import numpy as np
from pyrep import PyRep
from manipulator import Manipulator
from dh_convert import DHModel
from os.path import dirname, join, abspath
import gym
from gym import spaces
from pyrep.objects.shape import Shape

DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi

GOAL = {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20],
        'hard': [0, 20, 0, 15, 0, 20, 0, 20, 0, 20],
        'super hard': [0, -50, 0, -50, 0, -50, 0, -20, 0, -10, 0, 10]}


class ManipulatorEnv(gym.Env):
    def __init__(self, env_config):
        super(ManipulatorEnv, self).__init__()
        self.max_angles_vel = env_config['max_angles_vel']
        self.distance_threshold = env_config['distance_threshold']
        self.reward_type = env_config['reward_type']
        self.num_joints = env_config['num_joints']
        self.goal_set = env_config['goal_set']
        self._max_episode_steps = env_config['max_episode_steps']
        self.collision_cnt = env_config['collision_cnt']
        self.headless_mode = env_config['headless_mode']

        self.state_dim = self.num_joints + 12  # EE_point_position, EE_point_vel, goal_position, base_position
        self.action_dim = self.num_joints // 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.j_ang_idx = range(self.num_joints // 2)
        self.j_vel_idx = range(self.num_joints // 2, self.num_joints)
        self.e_pos_idx = range(self.num_joints, self.num_joints + 3)
        self.e_vel_idx = range(self.num_joints + 3, self.num_joints + 6)
        self.g_pos_idx = range(self.num_joints + 6, self.num_joints + 9)
        self.b_pos_idx = range(self.num_joints + 9, self.num_joints + 12)
        self.dh_model = DHModel(self.num_joints)
        self.observation = np.zeros((self.state_dim,))
        self.last_obs = None
        self._elapsed_steps = None

        self.pr = PyRep()
        self.pr.launch(join(dirname(abspath(__file__)), env_config['scene_file']), headless=self.headless_mode)
        self.pr.start()
        self.agent = Manipulator(num_joints=self.num_joints,
                                 collision_cnt=self.collision_cnt)
        # self.agent.set_control_loop_enabled(False)
        # self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.agent_target = Shape("target")
        self.agent_base = self.agent.get_base()
        self.initial_joint_positions = self.agent.get_joint_initial_positions()

    def _sample_goal(self):
        if self.goal_set in ['easy', 'hard', 'super hard']:
            theta = np.asarray(GOAL[self.goal_set]) * DEG2RAD
        elif self.goal_set == 'random':
                theta = np.vstack((np.zeros((self.action_dim,)),
                                   45 * DEG2RAD * np.random.uniform(low=-1, high=1,
                                                                    size=(self.action_dim,)))).T.flatten()
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
        self._elapsed_steps = 0
        self.goal_theta, self.goal, self.max_rewards = self._sample_goal()
        self.observation[self.g_pos_idx] = np.asarray(self.goal)
        self.observation[self.b_pos_idx] = np.asarray([0.2, 0, 1])
        self.agent_target.set_position(self.goal)
        self.agent.set_joint_positions(self.initial_joint_positions)
        observation, _ = self._get_state()
        self.last_obs = observation
        return observation

    def step(self, action):
        assert self._elapsed_steps is not None
        self._elapsed_steps += 1
        action = self.max_angles_vel * DEG2RAD * np.asarray(action)
        action = action[:, np.newaxis]
        action = np.concatenate((np.zeros((self.action_dim, 1)), action), axis=-1).flatten()
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        time_c = time.time()
        self.pr.step()  # Step the physics simulation
        observation, info = self._get_state()
        reward = self.cal_reward(observation[self.e_pos_idx], self.goal)
        done = np.linalg.norm(observation[self.e_pos_idx] - self.goal, axis=-1) <= self.distance_threshold
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        if info['collision_state']:
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
        'goal_set': 'super hard',
        'max_episode_steps': 100,
        'cc_model': False,
        'collision_cnt': 15,
        'scene_file': 'by_12_1.ttt',
        'headless_mode': True,
    }
    env = ManipulatorEnv(env_config)
    print('env created success')
    action_ = [1, 1, 1, -1, -1, -1]
    # action_ = np.random.uniform(-1, 1, size=(6,))
    time_a = time.time()
    lines = []
    for i in range(1):
        step = 0
        line = []
        obs = env.reset()
        while True:
            # time_a = time.time()
            obs, reward, done, info = env.step(action_)
            line.append(obs[0])
            # time_b = time.time()
            # print(time_b - time_a)
            step += 1
            print(step)
            # print(info['collision_state'])
            # if any(info['collision_state']):
            #     vrep.simxAddStatusbarMessage(env.clientID, 'collision detected', vrep.simx_opmode_oneshot)
            #     print('collision detected')
            #
            #     time.sleep(100)
            if done:
                time.sleep(5)
                if info['collision_state']:
                    print(f'collision detected in step {step}')
                lines.append(line)
                step = 0
                break
    # from matplotlib import pyplot as plt
    # for i in range(1):
    #     plt.plot(lines[i])
    # plt.show()
    time_b = time.time()
    print(time_b - time_a)
    env.end_simulation()
