import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
import os
import sys
main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(main_dir)
# sys.path.append(os.path.join(main_dir, 'vrep_pyrep'))
sys.path.append(os.path.join(main_dir, 'mujoco'))
from mujoco.env import ManipulatorEnv
import visdom


# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    # model_path = 'saved_models/her_9/1488.pt'
    model_path = 'saved_models/her_6/9800.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env_config = {
        'distance_threshold': args.distance_threshold,
        'reward_type': args.reward_type,
        'max_angles_vel': args.max_angles_vel,  # 10degree/s
        'num_joints': args.num_joints,
        'num_segments': args.num_segments,
        'cc_model': args.cc_model,
        'plane_model': args.plane_model,
        'goal_set': args.goal_set,
        'max_episode_steps': args.max_episode_steps,
        'collision_cnt': args.collision_cnt,
        'headless_mode': args.headless_mode,
        'scene_file': args.scene_file,
        'n_substeps': 100,
        'random_initial_state': args.random_initial_state,
    }
    env = ManipulatorEnv(env_config)
    env.action_space.seed(args.seed)
    env.seed()
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network

    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    goal_list = []
    achieved_goal_list = []
    for i in range(args.demo_length):
        observation = env.reset()
        # env.render()
        goal_list.append(observation['desired_goal'])
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        length = 0
        for t in range(env._max_episode_steps):
            length += 1
            if not args.headless_mode:
                env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, done, info = env.step(action)
            # print(observation_new['achieved_goal'])
            achieved_goal_list.append(observation_new['achieved_goal'])
            obs = observation_new['observation']
            if done:
                break
        print('the episode is: {}, length is {}, is success: {}'.format(i, length, info['is_success']))
    goal_list = np.array(goal_list)
    achieved_goal_list = np.array(achieved_goal_list)
    viz = visdom.Visdom(port=6016, env=args.code_version)
    viz.scatter(
        X=np.vstack([goal_list, achieved_goal_list]),
        Y=np.hstack([np.ones((len(goal_list),)), 2 * np.ones((len(achieved_goal_list,)))]),
        win='path',
        opts={
            'title': 'path',
            'legend': ['goal_path', 'achieved_path'],
            'markersize': 5
        }
    )
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(goal_list[:, 0], goal_list[:, 1], goal_list[:, 2], label='goal path')
    ax.plot(achieved_goal_list[:, 0], achieved_goal_list[:, 1], achieved_goal_list[:, 2], label='achieved path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

