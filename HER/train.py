import numpy as np
import gym
import os, sys
from arguments import get_args
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(main_dir)
# sys.path.append(os.path.join(main_dir, 'vrep_pyrep'))
sys.path.append(os.path.join(main_dir, 'mujoco'))

# from vrep_pyrep.env import ManipulatorEnv
from mujoco.env import ManipulatorEnv
from mujoco.env_test import EnvTest



def get_logger(code_version):
    import time
    import logging
    import os
    import sys

    main_dir = os.path.abspath(os.path.dirname(__file__))
    log_dir = os.path.join(main_dir, 'log')
    sub_log_dir = os.path.join(log_dir, sys.argv[0].split('.')[0])
    os.makedirs(sub_log_dir, exist_ok=True)
    log_name = code_version
    file_name = os.path.join(sub_log_dir, log_name + '.log')
    if os.path.exists(file_name):
        try:
            os.remove(file_name)
        except:
            pass

    logger = logging.getLogger('mani')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s [%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_env_params(env, goal_set):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    # # create the ddpg_agent
    # env = gym.make(args.env_name)
    # # set random seeds for reproduce
    # env.seed(args.seed)
    # set random seeds for reproduce
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    reg = ''
    for key, value in vars(args).items():
        reg += str(key) + ': ' + str(value) + '\n'
    logger.info(reg)
    env_config = {
        'distance_threshold': args.distance_threshold,
        'reward_type': args.reward_type,
        'max_angles_vel': args.max_angles_vel,  # 10degree/s
        'num_joints': args.num_joints,
        'num_segments': args.num_segments,
        'cc_model': args.cc_model,
        'plane_model': args.plane_model,
        'goal_set': args.goal_set,
        'eval_goal_set': args.eval_goal_set,
        'max_episode_steps': args.max_episode_steps,
        'collision_cnt': args.collision_cnt,
        'headless_mode': args.headless_mode,
        'scene_file': args.scene_file,
        'n_substeps': 100,
        'random_initial_state': args.random_initial_state,
        'add_ta': False,
        'add_peb': False,
        'is_her': True,
        'max_reset_period': args.max_reset_period,
        'reset_change_point': args.reset_change_point,
        'reset_change_period': args.reset_change_period,
    }
    if args.env_name == 'mani':
        env_name = ManipulatorEnv
    elif args.env_name == 'test':
        env_name = EnvTest
    env = env_name(env_config)
    env.action_space.seed(args.seed)
    # env.seed()

    # get the environment parameters
    env_params = get_env_params(env, args.goal_set)
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    if args.train:
        ddpg_trainer.learn()
    else:
        ddpg_trainer.eval()


if __name__ == '__main__':
    # take the configuration for the HER
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    logger = get_logger(args.code_version)
    launch(args)
