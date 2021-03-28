import torch
from rl_modules.models import Actor, ActorDense, ActorDenseSimple, ActorDenseASF
from arguments import get_args
import gym
import numpy as np
import random
import os
import sys
main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(main_dir)
# sys.path.append(os.path.join(main_dir, 'vrep_pyrep'))
sys.path.append(os.path.join(main_dir, 'mujoco'))
from mujoco.env import ManipulatorEnv
from mujoco.env_test import EnvTest
import visdom
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from multiprocessing import Process, Queue
from mujoco_py.mjviewer import save_video

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


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


if __name__ == '__main__':
    args = get_args()
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
        'add_dtt': args.add_dtt,
        'max_reset_period': args.max_reset_period,
        'reset_change_point': args.reset_change_point,
        'reset_change_period': args.reset_change_period,
        'fixed_reset': args.fixed_reset,
    }
    if args.env_name == 'mani':
        env_name = ManipulatorEnv
    elif args.env_name == 'test':
        env_name = EnvTest
    env = env_name(env_config)
    # env.action_space.seed(args.seed)
    # env.seed()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    observation = env.reset()
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    if args.actor_type == 'normal':
        actor = Actor
    elif args.actor_type == 'dense':
        actor = ActorDense
    elif args.actor_type == 'dense_simple':
        actor = ActorDenseSimple
    elif args.actor_type == 'dense_asf':
        actor = ActorDenseASF
    else:
        raise ValueError

    vis = visdom.Visdom(port=6016, env=args.code_version)
    vis.line(X=[0], Y=[0], win='result', opts=dict(Xlabel='episode', Ylabel='result', title='result'))
    vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
    vis.line(X=[0], Y=[0], win='success rate',
             opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='success rate'))
    vis.scatter(
        X=np.array([[1.4, 0, 1]]),
        Y=[2],
        win='cnt',
        opts={
            'title': 'if success',
            'legend': ['fail', 'success'],
            'markersize': 5
        }
    )

    from _collections import deque
    result_queue = deque(maxlen=20)

    goal_list = []
    achieved_path = []
    result_list = []

    actor_network = actor(args, env_params)
    # model_path = '/media/cq/000CF0AE00072D66/saved_models/her_17/4188.pt'
    # model_path = 'saved_models/her_32/10000.pt'
    # model_path = 'saved_models/her_111/model.pt'
    # model_path = '/media/cq/000CF0AE00072D66/saved_models/her_46/model.pt'
    file_list = [
                 # 'saved_models/her_32/10000.pt',
        'saved_models/her_80/999.pt',
        # 'saved_models/her_112/model.pt',
                 ]
    dice_list = [[] for _ in range(len(file_list))]
    path_len_list = [[] for _ in range(len(file_list))]
    for file_list_ in range(len(file_list)):
        env.action_space.seed(args.seed)
        env.seed()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        o_mean, o_std, g_mean, g_std, model = torch.load(file_list[file_list_], map_location=lambda storage, loc: storage)
        actor_network.load_state_dict(model)
        actor_network.eval()
        for i in range(args.demo_length):
            observation = env.reset(eval=True)
            achieved_path.append(observation['achieved_goal'])
            # env.render()
            goal_list.append(observation['desired_goal'])
            # print(observation['desired_goal'])
            # start to do the demo
            obs = observation['observation']
            g = observation['desired_goal']
            length = 0
            result = 0
            dice = 0
            for t in range(env._max_episode_steps):
                length += 1
                if not args.headless_mode:
                    env.render()
                inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
                with torch.no_grad():
                    pi = actor_network(inputs.unsqueeze(0))
                action = pi.detach().numpy().squeeze()
                # noise = np.random.normal(0, 0.05, size=action.shape)
                # # Add noise to the action for exploration
                # action = (action + noise).clip(env.action_space.low, env.action_space.high)

                # put actions into the environment
                observation_new, reward, done, info = env.step(action)
                next_joint_state = observation_new['observation'][env.j_ang_idx]
                dice += np.absolute(np.concatenate([
                    next_joint_state[range(0, env.action_dim - 2, 2)] - next_joint_state[range(2, env.action_dim, 2)],
                    next_joint_state[range(1, env.action_dim - 2, 2)] - next_joint_state[range(3, env.action_dim, 2)]
                ], axis=-1)).sum(axis=-1)
                # print(observation_new['achieved_goal'])
                achieved_path.append(observation_new['achieved_goal'])
                obs = observation_new['observation']
                if done:
                    if not args.headless_mode:
                        env.render()
                    # if done and path_length < args.max_episode_steps and not any(info['collision_state']):
                    if done and length < args.max_episode_steps:
                        result = 1
                    break
            path_len_list[file_list_].append(length)
            dice_list[file_list_].append(dice)
            result_list.append(result + 1)
            result_queue.append(result)
            eval_success_rate = sum(result_queue) / len(result_queue)
            vis.line(X=[i], Y=[result], win='result', update='append')
            vis.line(X=[i], Y=[length], win='path len', update='append')
            vis.line(X=[i], Y=[eval_success_rate * 100], win='success rate', update='append')
            print('the episode is: {}, length is {}, is success: {}'.format(i, length, info['is_success']))


    # fps = (1 / env.viewer._time_per_render)
    # env.viewer._video_process = Process(target=save_video,
    #                               args=(env.viewer._video_queue, env.viewer._video_path % env.viewer._video_idx, fps))
    # env.viewer._video_process.start()
    #
    # time.sleep(4)
    # env.viewer._video_queue.put(None)
    # env.viewer._video_process.join()

    # print(np.array(result_list).sum())
    env.close()

    plt.show()
    # 目标达成情况
    result_list = np.array(result_list)
    goal_list = np.array(goal_list)
    vis.scatter(
        X=goal_list,
        Y=result_list,
        win='if success',
        opts={
            'title': 'if success',
            'legend': ['fail', 'success'],
            'markersize': 5
        }
    )
    # 示教路线和实际路线
    goal_list = np.array(goal_list)
    achieved_path = np.array(achieved_path)
    vis.scatter(
        X=np.vstack([goal_list, achieved_path]),
        Y=np.hstack([np.ones((len(goal_list),)), 2 * np.ones((len(achieved_path, )))]),
        win='path',
        opts={
            'title': 'path',
            'legend': ['goal_path', 'achieved_path'],
            'markersize': 5
        }
    )
    # matplotlib plot
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(goal_list[:, 0], goal_list[:, 1], goal_list[:, 2], label='goal path')
    ax.plot(achieved_path[:, 0], achieved_path[:, 1], achieved_path[:, 2], label='achieved path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.legend()
    plt.savefig('/home/cq/code/manipulator/plot/saved_fig/her_eval_on_gate_2.png', bbox_inches='tight')
    plt.show()

