from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
from baselines import logger
from env import ManipulatorEnv

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """
    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)

class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """
    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        logger.warn('Render not defined for %s'%self)

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self):
        self.venv.render()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

import numpy as np
from gym import spaces


def worker(index, remote, parent_remote, env_fn_wrapper, env_config):
    parent_remote.close()
    env = env_fn_wrapper.x(env_config)
    env.action_space.seed(index)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(**data)
            remote.send(ob)
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv, ManipulatorEnv):
    def __init__(self, env_fns, env_config):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(index, work_remote, remote, CloudpickleWrapper(env_fn), env_config))
            for index, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns))]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.max_angles_vel = env_config['max_angles_vel']
        self.distance_threshold = env_config['distance_threshold']
        self.reward_type = env_config['reward_type']
        self.num_joints = env_config['num_joints']
        self.num_segments = env_config['num_segments']
        self.cc_model = env_config['cc_model']
        self.plane_model = env_config['plane_model']
        self.goal_set = env_config['goal_set']
        self.eval_goal_set = env_config['eval_goal_set']
        self._max_episode_steps = env_config['max_episode_steps']
        self.collision_cnt = env_config['collision_cnt']
        self.headless_mode = env_config['headless_mode']
        self.random_initial_state = env_config.get('random_initial_state', False)
        self.add_peb = env_config['add_peb']
        self.add_dtt = env_config['add_dtt']
        self.is_her = env_config['is_her']
        self.max_reset_period = env_config['max_reset_period']
        self.reset_change_period = env_config['reset_change_period']
        self.reset_change_point = env_config['reset_change_point']
        self.fixed_reset = env_config['fixed_reset']

    def step(self, actions):
        return VecEnv.step(self, actions)

    def process_obs(self, recv_obs):
        obs = list(zip(*[obs.values() for obs in recv_obs]))
        stacked_obs = {key: np.vstack(obs[i]) for i, key in enumerate(recv_obs[0].keys())}
        return stacked_obs

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        infos = {'is_success': np.stack(info['is_success'] for info in infos)}
        return self.process_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, **keyargs):
        for remote in self.remotes:
            remote.send(('reset', keyargs))
        recv_obs = [remote.recv() for remote in self.remotes]
        stacked_obs = self.process_obs(recv_obs)
        return stacked_obs

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    # def render(self, mode='human'):
    #     for pipe in self.remotes:
    #         pipe.send(('render', None))
    #     imgs = [pipe.recv() for pipe in self.remotes]
    #     bigimg = tile_images(imgs)
    #     if mode == 'human':
    #         import cv2
    #         cv2.imshow('vecenv', bigimg[:,:,::-1])
    #         cv2.waitKey(1)
    #     elif mode == 'rgb_array':
    #         return bigimg
    #     else:
    #         raise NotImplementedError




if __name__ == '__main__':
    goal_index = {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20],
                  'hard': [20, 20, 15, 15, 20, 20, 20, 20, 20, 20, -10, -10, 20, 20, 15, 15, 20, 20, 20, 20, 20, 20, -10, -10],
                  'super hard': [0, -50, 0, -50, 0, -50, 0, -20, 0, -10]}
    env_config = {
        'distance_threshold': 0.02,
        'reward_type': 'dense distance',
        'max_angles_vel': 10,  # 10degree/s
        'num_joints': 12,
        'num_segments': 2,
        'cc_model': False,
        'plane_model': False,
        'goal_set': 'block0',
        'eval_goal_set': 'block0',
        'max_episode_steps': 100,
        'collision_cnt': 15,
        'scene_file': 'mani_block0_1_env_6.xml',
        'headless_mode': False,
        'n_substeps': 100,
        'random_initial_state': False,
        'add_ta': False,
        'add_peb': False,
        'add_dtt': True,
        'is_her': True,
        'max_reset_period': 10,
        'reset_change_point': 20,
        'reset_change_period': 30,
        'fixed_reset': False,
    }

    # env_fans = [ManipulatorEnv for i in range(2)]
    # env = SubprocVecEnv(env_fans, env_config)
    # obs = env.reset()
    # print(obs)
    # obs, reward, done, info = env.step(np.ones((4, 12)))
    # print(obs, reward, done, info)
    # for i in range(100):
    #     action = np.ones((4, 12))
    #     obs, reward, done, info = env.step(action)
    #     print(obs)

    a = np.ones((2,3,4,5)).swapaxes(1,2)
    a = a.reshape(a.shape[0]*a.shape[1], *a.shape[2:])

    print(a.shape)