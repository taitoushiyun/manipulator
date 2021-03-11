import importmagic
from mujoco.env import ManipulatorEnv
from TD3.td3_agent import TD3Agent, td3_torcs
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
        'max_episode_steps': args_.max_episode_steps,
        'collision_cnt': args_.collision_cnt,
        'headless_mode': args_.headless_mode,
        'scene_file': args_.scene_file,
        'n_substeps': args_.n_substeps,
        'random_initial_state': args_.random_initial_state,
        'add_ta': args_.add_ta,
        'add_peb': args_.add_peb,
        'is_her': args_.is_her,
    }
    env = ManipulatorEnv(env_config)
    env.action_space.seed(args_.seed)
    # env = gym.make('LunarLanderContinuous-v2')
    obs = env.reset(args_.goal_set)
    agent = TD3Agent(args=args_,
                     state_size=obs['observation'].shape[0] + obs['desired_goal'].shape[0],
                     action_size=env.action_space.shape[0],
                     max_action=env.action_space.high,
                     min_action=env.action_space.low,
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
                    'title': 'if success',
                    'legend': ['fail', 'success'],
                    'markersize': 5
                }
            )
            from _collections import deque
            result_queue = deque(maxlen=20)
            goal_list = []
            result_list = []
            model = torch.load(
                f'/home/cq/code/manipulator/TD3/checkpoints/td3_122/1830.pth')  # 'PPO/checkpoints/40.pth'
            # f'/media/cq/系统/Users/Administrator/Desktop/实验记录/td3_14/checkpoints/actor/1000.pth')
            agent.actor_local.load_state_dict(model)

            for i in range(2000):
                state = env.reset(args_.eval_goal_set)
                goal_list.append(state['desired_goal'])
                state = np.concatenate([state['observation'], state['desired_goal']])
                total_reward = 0
                path_length = 0
                for _ in range(args_.max_episode_steps):
                    if not args_.headless_mode:
                        env.render()
                    action = agent.act(state, episode_step=i)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.concatenate([next_state['observation'], next_state['desired_goal']])
                    total_reward += reward
                    path_length += 1
                    state = next_state
                    if args_.add_peb:
                        if done or i == args_.max_episode_steps - 1:
                            result = 0.
                            if done:
                                result = 1.
                            break
                    else:
                        if done:
                            # env.render()
                            result = 0.
                            if done and path_length < args_.max_episode_steps:
                                # if done and total_len < max_episode_length and not any(info['collision_state']):
                                result = 1.
                            break
                result_list.append(result+1)
                result_queue.append(result)
                eval_success_rate = sum(result_queue) / len(result_queue)
                print(f'episode {i} result {result} path len {path_length}')
                vis.line(X=[i], Y=[result], win='result', update='append')
                vis.line(X=[i], Y=[path_length], win='path len', update='append')
                vis.line(X=[i], Y=[eval_success_rate * 100], win='success rate', update='append')
                # vis.scatter(
                #     X=goal_list,
                #     Y=result_list,
                #     win='cnt',
                #     opts={
                #         'title': 'if success',
                #         'legend': ['fail', 'success'],
                #         'markersize': 5
                #     }
                # )
            result_list = np.array(result_list)
            goal_list = np.array(goal_list)
            from matplotlib import pyplot as plt
            vis.scatter(
                X=goal_list,
                Y=result_list,
                opts={
                    'title': 'if success',
                    'legend': ['fail', 'success'],
                    'markersize': 5
                }
            )


    finally:
        # env.end_simulation()  # This is for shutting down TORCS
        print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 for manipulator.')
    parser.add_argument('--code-version', type=str, default='td3_121')
    parser.add_argument('--vis-port', type=int, default=6016)
    parser.add_argument('--seed', type=int, default=1)
    #  TD3 config
    parser.add_argument('--actor-hidden', type=list, default=[128, 128])
    parser.add_argument('--critic-hidden', type=list, default=[64, 64])
    parser.add_argument('--buffer-size', type=int, default=int(1e7))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.6)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--lr-actor', type=float, default=1e-3)
    parser.add_argument('--lr-critic', type=float, default=1e-3)
    parser.add_argument('--update-every-step', type=int, default=2)
    parser.add_argument('--random-start', type=int, default=2000)
    parser.add_argument('--noise-decay-period', type=float, default=500.)
    parser.add_argument('--n-test-rollouts', type=int, default=10)
    parser.add_argument('--test-interval', type=int, default=20)
    # env config
    parser.add_argument('--max-episode-steps', type=int, default=100)
    parser.add_argument('--distance-threshold', type=float, default=0.02)
    parser.add_argument('--reward-type', type=str, default='dense potential')
    parser.add_argument('--max-angles-vel', type=float, default=10.)
    parser.add_argument('--num-joints', type=int, default=24)
    parser.add_argument('--num-segments', type=int, default=2)
    parser.add_argument('--plane-model', action='store_true')
    parser.add_argument('--cc-model', action='store_true')
    parser.add_argument('--goal-set', type=str, choices=['easy', 'hard', 'super hard', 'random',
                                                         'draw0'],
                        default='hard')
    parser.add_argument('--eval-goal-set', type=str, default='hard')
    parser.add_argument('--collision-cnt', type=int, default=27)
    parser.add_argument('--scene-file', type=str, default='mani_env_24.xml')
    parser.add_argument('--headless-mode', action='store_true')
    parser.add_argument('--n-substeps', type=int, default=100)
    parser.add_argument('--random-initial-state', action='store_true')
    parser.add_argument('--add-ta', action='store_true')
    parser.add_argument('--add-peb', action='store_true')
    parser.add_argument('--is_her', type=bool, default=False)


    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=2000)

    args = parser.parse_args()
    # write the selected car to configuration file
    logger = get_logger(args.code_version)
    playGame(args, args.train, args.episodes)




