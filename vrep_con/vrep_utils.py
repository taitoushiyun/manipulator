import time
import numpy as np
import copy
from vrep_con import vrep
from vrep_con.vrep_config import VREP_Config
from vrep_con.dh_convert import DHModel
import gym
from gym import spaces

DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi


class ManipulatorEnv(gym.Env):
    def __init__(self, index):
        super(ManipulatorEnv, self).__init__()
        config = copy.deepcopy(VREP_Config)
        self.port = 20000 + index
        self.handles = {}
        self.clientID = None
        self.running = False
        self.connected = False

        self.step_cnt = 0
        self.num_episodes = 0
        self.T = config['T']
        self.max_torque = config['max_torque']
        self.max_angles_vel = config['max_angles_vel']
        self.distance_threshold = config['distance_threshold']
        self.reward_type = config['reward_type']
        self.buffer_type = config['buffer_type']

        self.num_joints = config['num_joints']
        self.state_dim = self.num_joints + 12       # EE_point_position, EE_point_vel, goal_position, base_position
        self.action_dim = self.num_joints // 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.j_ang_idx = range(self.num_joints // 2)
        self.j_vel_idx = range(self.num_joints // 2, self.num_joints)
        self.e_pos_idx = range(self.num_joints,  self.num_joints + 3)
        self.e_vel_idx = range(self.num_joints + 3, self.num_joints + 6)
        self.g_pos_idx = range(self.num_joints + 6, self.num_joints + 9)
        self.b_pos_idx = range(self.num_joints + 9, self.num_joints + 12)
        self.dh_model = DHModel(index, self.num_joints)
        self.goal_theta, self.goal = self._sample_goal()
        self.observation = np.zeros((self.state_dim, ))
        self.observation[self.g_pos_idx] = np.asarray(self.goal)
        self.observation[self.b_pos_idx] = np.asarray([0.2, -index, 1])
        self._init_vrep()
        self._elapsed_steps = None
        self._max_episode_steps = 100

    def _init_vrep(self):
        vrep.simxFinish(-1)
        attempts = 0
        while True:
            self.clientID = vrep.simxStart('127.0.0.1', self.port, True, True, 500, 5)
            attempts += 1
            if self.clientID != -1:
                self.connected = True
                break
            elif attempts < 10:
                print('Unable to connect to V-REP, Retrying...')
                time.sleep(0.2)
            else:
                raise RuntimeError('Unable to connect to V-REP.')
        print('Connection success')
        self.get_handles()

    def get_handles(self):
        joint_ids = [f'joint{idx_0}_{idx_1}#{self.port - 20000}'
                     for idx_0 in range(self.num_joints // 2) for idx_1 in range(2)]
        point_ids = [f'aux{self.num_joints // 2}#{self.port-20000}', f'goal#{self.port-20000}', f'base#{self.port-20000}']
        joint_handles = [vrep.simxGetObjectHandle(self.clientID, joint_id, vrep.simx_opmode_blocking)[1]
                         for joint_id in joint_ids]
        point_handles = [vrep.simxGetObjectHandle(self.clientID, point_id, vrep.simx_opmode_blocking)[1]
                         for point_id in point_ids]
        self.handles['joint'] = joint_handles
        self.handles['point'] = point_handles

    def _sample_goal(self):
        # theta = np.random.randn(self.num_joints)
        theta = np.asarray([0, -50, 0, -50, 0, -50, 0, 0, -20, -10]) * DEG2RAD
        goal_theta = np.clip(theta, -3, 3)
        print(f'goal sample for joints is {goal_theta}')
        goal = self.dh_model.forward_kinematics(goal_theta)
        print(f'goal sample for end point is {goal}')
        return goal_theta, goal

    def end_simulation(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        vrep.simxGetPingTime(self.clientID)
        vrep.simxFinish(self.clientID)

    def stop_sim(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        try:
            while True:
                vrep.simxGetIntegerSignal(self.clientID, 'sig_debug', vrep.simx_opmode_blocking)
                e = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
                still_running = e[1] & 1
                if not still_running:
                    break
        except:
            print('pass')
            pass
        self.running = False

    def reset(self):
        self._elapsed_steps = 0
        if self.running:
            self.stop_sim()
        # set initial state
        vrep.simxSetObjectPosition(self.clientID, self.handles['point'][1],
                                   -1, self.goal, vrep.simx_opmode_oneshot)
        vrep.simxSynchronous(self.clientID, True)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        self.running = True
        # start fetching streaming data
        for i in range(self.num_joints // 2):
            vrep.simxGetJointPosition(self.clientID, self.handles['joint'][2 * i + 1], vrep.simx_opmode_streaming)
        for i in range(self.num_joints // 2):
            vrep.simxGetObjectFloatParameter(self.clientID, self.handles['joint'][2 * i + 1], 2012, vrep.simx_opmode_streaming)
        vrep.simxGetObjectPosition(self.clientID, self.handles['point'][0], -1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectVelocity(self.clientID, self.handles['point'][0], vrep.simx_opmode_streaming)
        vrep.simxGetObjectPosition(self.clientID, self.handles['point'][1], -1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectPosition(self.clientID, self.handles['point'][2], -1, vrep.simx_opmode_streaming)
        # first synchronous step
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        self.step_cnt = 0
        self.running = True
        observation = self.get_state(vrep.simx_opmode_buffer)
        return observation

    def step(self, action):
        assert self._elapsed_steps is not None
        self._elapsed_steps += 1
        action = self.max_angles_vel * DEG2RAD * np.asarray(action)
        action = action[:, np.newaxis]
        action = np.concatenate((np.zeros((self.action_dim, 1)), action), axis=-1).flatten()
        self.set_joint_effect(action)
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        observation = self.get_state(vrep.simx_opmode_buffer)
        reward = self.cal_reward(observation[self.e_pos_idx], self.goal)
        done = self.cal_termination(observation[self.e_pos_idx], self.goal)
        info = {}
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def cal_reward(self, achieved_goal, goal):
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def cal_termination(self, achieved_goal, goal):
        d = self.goal_distance(achieved_goal, goal)
        return d <= self.distance_threshold

    def set_joint_effect(self, action):
        assert len(action) == self.num_joints, 'action dimension wrong'
        vrep.simxPauseCommunication(self.clientID, True)
        for i in range(self.num_joints):
            vrep.simxSetJointTargetVelocity(self.clientID, self.handles['joint'][i],
                                            action[i], vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.clientID, False)

    def get_state(self, mode):
        state = np.zeros(self.state_dim)
        jas = [vrep.simxGetJointPosition(self.clientID, self.handles['joint'][2*i+1], mode)[1]
               for i in range(self.num_joints // 2)]
        jvs = [vrep.simxGetObjectFloatParameter(self.clientID, self.handles['joint'][2*i+1], 2012, mode)[1]
               for i in range(self.num_joints // 2)]
        ee_point_pos = vrep.simxGetObjectPosition(self.clientID, self.handles['point'][0],
                                                  -1, mode)[1]
        ee_point_vel = vrep.simxGetObjectVelocity(self.clientID, self.handles['point'][0],
                                                  mode)[1]
        state[self.j_ang_idx] = np.asarray(jas) * RAD2DEG
        state[self.j_vel_idx] = np.asarray(jvs) * RAD2DEG
        state[self.e_pos_idx] = np.asarray(ee_point_pos)
        state[self.e_vel_idx] = np.asarray(ee_point_vel)
        state[self.g_pos_idx] = self.observation[self.g_pos_idx]
        state[self.b_pos_idx] = self.observation[self.b_pos_idx]
        return state


if __name__ == '__main__':
    env = ManipulatorEnv(0)
    print('env created success')
    obs = env.reset()
    print('reset success')
    action = [0, 45, 0, -45, 0, -45, 0, 45, 0, 0]
    vrep.simxPauseCommunication(env.clientID, True)
    for i in range(env.num_joints):
        vrep.simxSetJointPosition(env.clientID, env.handles['joint'][i],
                                        action[i]*DEG2RAD, vrep.simx_opmode_oneshot)
    vrep.simxPauseCommunication(env.clientID, False)
    time.sleep(100)





