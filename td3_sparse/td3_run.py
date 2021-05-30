import importmagic
from mujoco.env import ManipulatorEnv
from mujoco.vec_env import SubprocVecEnv
from mujoco.env_test import EnvTest
from td3_sparse.td3_agent import TD3Agent, td3_torcs
import visdom
import numpy as np
import random
import argparse
import subprocess
import time
import os
import torch
from tqdm import tqdm
import json
from itertools import count
import gym


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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


def playGame(args_, train=True, episode_count=2000):
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    reg = ''
    for key, value in vars(args).items():
        reg += str(key) + ': ' + str(value) + '\n'
    logger.info(reg)
    env_config = {
        'distance_threshold': args_.distance_threshold,
        'reward_type': args_.reward_type,
        'max_angles_vel': args_.max_angles_vel,  # 10degree/s
        'num_joints': args_.num_joints,
        'num_segments': args_.num_segments,
        'cc_model': args_.cc_model,
        'plane_model': args_.plane_model,
        'goal_set': args_.goal_set,
        'eval_goal_set': args_.eval_goal_set,
        'max_episode_steps': args_.max_episode_steps,
        'collision_cnt': args_.collision_cnt,
        'headless_mode': args_.headless_mode,
        'scene_file': args_.scene_file,
        'n_substeps': args_.n_substeps,
        'random_initial_state': args_.random_initial_state,
        'add_ta': args_.add_ta,
        'add_peb': args_.add_peb,
        'is_her': args_.is_her,
        'add_dtt': args_.add_dtt,
        'max_reset_period': args_.max_reset_period,
        'reset_change_point': args_.reset_change_point,
        'reset_change_period': args_.reset_change_period,
        'fixed_reset': args_.fixed_reset,
        'nenvs': args_.nenvs,
    }

    env_fans = [ManipulatorEnv for _ in range(args.nenvs)]
    env = SubprocVecEnv(env_fans, env_config)

    # if args.env_name == 'mani':
    #     env_name = ManipulatorEnv
    # elif args.env_name == 'test':
    #     env_name = EnvTest
    # env = env_name(env_config)
    # env.action_space.seed(args_.seed)

    obs = env.reset()
    agent = TD3Agent(args=args_,
                     env=env,
                     state_size=obs['observation'].shape[-1],
                     action_size=env.action_space.shape[-1],
                     max_action=env.action_space.high.flatten()[0],
                     min_action=env.action_space.low.flatten()[0],
                     actor_hidden=args_.actor_hidden,
                     critic_hidden=args_.critic_hidden,
                     random_seed=0,
                     gamma=args_.gamma,
                     tau=args_.tau,
                     lr_actor=args_.lr_actor,
                     lr_critic=args_.lr_critic,
                     update_every_step=args_.update_every_step,
                     random_start=args_.random_start,
                     noise_drop_rate=args_.noise_decay_period)

    try:
        # try:
        #     agent.actor_local.load_state_dict(torch.load(os.path.join(model_dir, 'actor.pth')))
        #     agent.critic_local1.load_state_dict(torch.load(os.path.join(model_dir, 'critic1.pth')))
        #     agent.critic_local2.load_state_dict(torch.load(os.path.join(model_dir, 'critic2.pth')))
        #     print("Weight load successfully")
        # except:
        #     print("Cannot find the weight")


        if train:
            vis = visdom.Visdom(port=args.vis_port, env=args.code_version)
            td3_torcs(env, agent, episode_count, args.max_episode_steps,
                      os.path.join('checkpoints', args_.code_version), vis, args_)
        else:
            vis = visdom.Visdom(port=args.vis_port, env=args.code_version)
            vis.line(X=[0], Y=[0], win='result', opts=dict(Xlabel='episode', Ylabel='result', title='result'))
            vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
            vis.line(X=[0], Y=[0], win='success rate',
                     opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='success rate'))
            vis.scatter(
                X=np.array([[0.8, 0, 1]]),
                Y=[2],
                win='cnt',
                opts={

                    'legend': ['fail', 'success'],
                    'markersize': 5
                }
            )
            from _collections import deque
            result_queue = deque(maxlen=20)
            goal_list = []
            result_list = []
            # file_list = [f'/home/cq/code/manipulator/TD3/checkpoints/td3_155/950.pth',
            #              f'/home/cq/code/manipulator/TD3/checkpoints/td3_156/990.pth',
            #              f'/home/cq/code/manipulator/TD3/checkpoints/td3_157/990.pth',
            #              f'/home/cq/code/manipulator/TD3/checkpoints/td3_185/950.pth']
            # file_list = [f'/home/cq/code/manipulator/TD3/checkpoints/td3_203/19990.pth']
            file_list = [f'/media/cq/新加卷/checkpoints/td3_checkpoints/td3_101/9999.pth']
            achieved_path = [[] for _ in range(len(file_list))]
            for k in range(len(file_list)):
                model = torch.load(file_list[k])  # 'PPO/checkpoints/40.pth'
                # f'/media/cq/系统/Users/Administrator/Desktop/实验记录/td3_14/checkpoints/actor/1000.pth')
                agent.actor_local.load_state_dict(model)

                for i in range(1000):

                    state = env.reset(eval=True)
                    achieved_path[k].append(state['achieved_goal'].copy())
                    goal_list.append(env.goal.copy())
                    state = np.concatenate([state['observation'], state['desired_goal']])
                    total_reward = 0
                    path_length = 0
                    joints_total = 0
                    for step in range(args_.max_episode_steps):
                        if not args_.headless_mode:
                            env.render()
                        action = agent.act(state, episode_step=args_.noise_decay_period + 1, add_noise=False)
                        next_state, reward, done, info = env.step(action)
                        achieved_path[k].append(next_state['achieved_goal'].copy())
                        next_state = np.concatenate([next_state['observation'], next_state['desired_goal']])
                        joints_total += np.absolute(next_state[env.j_ang_idx]).sum()
                        total_reward += reward
                        path_length += 1
                        state = next_state
                        if args_.add_peb:
                            if done or path_length == args_.max_episode_steps:
                                if not args_.headless_mode:
                                    env.render()
                                result = 0.
                                if done:
                                    result = 1.
                                break
                        else:
                            if done:
                                if not args_.headless_mode:
                                    env.render()
                                result = 0.
                                if done and path_length < args_.max_episode_steps:
                                    # if done and total_len < max_episode_length and not any(info['collision_state']):
                                    result = 1.
                                break
                    # if i % 10 == 0:
                    #     print(joints_total)
                    #     joints_total = 0


                    result_list.append(result+1)
                    result_queue.append(result)
                    eval_success_rate = sum(result_queue) / len(result_queue)
                    print(f'episode {i} result {result} path len {path_length}')
                    vis.line(X=[i], Y=[result], win='result', update='append')
                    vis.line(X=[i], Y=[path_length], win='path len', update='append')
                    vis.line(X=[i], Y=[eval_success_rate * 100], win='success rate', update='append')
                achieved_path[k] = np.array(achieved_path[k])
            result_list = np.array(result_list)
            goal_list = np.array(goal_list)
            # print(result_list.sum())
            from matplotlib import pyplot as plt
            import matplotlib as mpl
            vis.scatter(
                X=goal_list,
                Y=result_list,
                opts={

                    'legend': ['fail', 'success'],
                    'markersize': 5
                }
            )
            # mpl.rcParams['legend.fontsize'] = 10
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # for i in range(len(file_list)):
            #     ax.scatter(achieved_path[i][:, 0], achieved_path[i][:, 1], achieved_path[i][:, 2], label='achieved path')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # set_axes_equal(ax)
            # ax.legend()
            # plt.show()



    finally:
        # env.end_simulation()  # This is for shutting down TORCS
        print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 for manipulator.')
    parser.add_argument('--code-version', type=str, default='block_5')
    parser.add_argument('--vis-port', type=int, default=6016)
    parser.add_argument('--seed', type=int, default=1)
    # unused
    parser.add_argument('--fixed-reset', action='store_true')
    parser.add_argument('--env-name', type=str, default='mani', help='the environment name')
    #  TD3 config
    parser.add_argument('--actor-hidden', type=list, default=[128, 128])
    parser.add_argument('--critic-hidden', type=list, default=[64, 64])
    parser.add_argument('--buffer-size', type=int, default=int(1e7))
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--lr-actor', type=float, default=1e-3)
    parser.add_argument('--lr-critic', type=float, default=1e-3)
    parser.add_argument('--update-every-step', type=int, default=2)
    parser.add_argument('--random-start', type=int, default=2000)
    parser.add_argument('--noise-decay-period', type=float, default=1000.)
    parser.add_argument('--n-test-rollouts', type=int, default=10)
    parser.add_argument('--test-interval', type=int, default=20)

    parser.add_argument('--action-q-ratio', type=float, default=0.1)
    parser.add_argument('--action-q', action='store_true')

    parser.add_argument('--actor-type', type=str, default='dense')
    parser.add_argument('--critic-type', type=str, default='dense')
    parser.add_argument('--curiosity-type', type=str, choices=['forward', 'rnd'], default='rnd')
    parser.add_argument('--rnd-net', type=str, choices=['mlp', 'densenet'], default='densenet')
    parser.add_argument('--use-popart', action='store_true')
    parser.add_argument('--use-rms-reward', action='store_true')
    parser.add_argument('--lr-critic-explore', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')  # TODO
    parser.add_argument('--lr-predict', type=float, default=0.001, help='the learning rate of the predict net')
    parser.add_argument('--q-explore-weight', type=float, default=0.1)
    # env config
    parser.add_argument('--nenvs', type=int, default=8)
    parser.add_argument('--max-episode-steps', type=int, default=50)
    parser.add_argument('--distance-threshold', type=float, default=0.02)
    parser.add_argument('--reward-type', type=str, default='sparse')
    parser.add_argument('--max-angles-vel', type=float, default=10.)
    parser.add_argument('--num-joints', type=int, default=12)
    parser.add_argument('--num-segments', type=int, default=2)
    parser.add_argument('--plane-model', action='store_true')
    parser.add_argument('--cc-model', action='store_true')
    parser.add_argument('--goal-set', type=str, default='hard')
    parser.add_argument('--eval-goal-set', type=str, default='hard')
    parser.add_argument('--collision-cnt', type=int, default=15)
    parser.add_argument('--scene-file', type=str, default='mani_env_6.xml')
    parser.add_argument('--headless-mode', action='store_true')
    parser.add_argument('--n-substeps', type=int, default=100)
    parser.add_argument('--random-initial-state', action='store_true')
    parser.add_argument('--add-ta', action='store_true')
    parser.add_argument('--add-peb', action='store_true')
    parser.add_argument('--add-dtt', action='store_true')
    parser.add_argument('--is-her', type=bool, default=False)
    parser.add_argument('--max-reset-period', type=int, default=10)
    parser.add_argument('--reset-change-point', type=int, default=0)
    parser.add_argument('--reset-change-period', type=int, default=30)


    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=10000)

    args = parser.parse_args()
    # write the selected car to configuration file
    logger = get_logger(args.code_version)
    playGame(args, args.train, args.episodes)




