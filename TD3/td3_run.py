import importmagic
from vrep_pyrep.env import ManipulatorEnv
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
from TD3.logger import logger


def playGame(args_, train=True, episode_count=2000):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
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
    }
    env = ManipulatorEnv(env_config)
    # env = gym.make('LunarLanderContinuous-v2')

    agent = TD3Agent(state_size=env.observation_space.shape[0],
                     action_size=env.action_space.shape[0],
                     max_action=env.action_space.high,
                     min_action=env.action_space.low, random_seed=0)

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
                      os.path.join('checkpoints', args_.code_version), vis)
        else:
            vis = visdom.Visdom(port=args.vis_port, env=args.code_version)
            vis.line(X=[0], Y=[0], win='result', opts=dict(Xlabel='episode', Ylabel='result', title='result'))
            vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
            vis.line(X=[0], Y=[0], win='success rate',
                     opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='success rate'))
            from _collections import deque
            result_queue = deque(maxlen=20)
            for i in range(100):
                # if i % 5 == 0:
                model = torch.load(
                    f'/home/cq/code/manipulator/TD3/checkpoints/td3_30/9999.pth')  # 'PPO/checkpoints/40.pth'
                    # f'/media/cq/系统/Users/Administrator/Desktop/实验记录/td3_18/checkpoints/actor/{i}.pth')
                agent.actor_local.load_state_dict(model)

                state = env.reset()
                total_reward = 0
                path_length = 0
                for t in count():
                    action = agent.act(state, add_noise=False)
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    path_length += 1
                    state = next_state
                    if done or path_length >= args.max_episode_steps:
                        # print(f"Episode length: {t+1}")
                        result = 0
                        if done and path_length < args.max_episode_steps and not any(info['collision_state']):
                            result = 1
                        break
                result_queue.append(result)
                eval_success_rate = sum(result_queue) / len(result_queue)
                print(f'episode {i} result {result} path len {path_length}')
                vis.line(X=[i], Y=[result], win='result', update='append')
                vis.line(X=[i], Y=[path_length], win='path len', update='append')
                vis.line(X=[i], Y=[eval_success_rate * 100], win='success rate', update='append')

    finally:
        env.end_simulation()  # This is for shutting down TORCS
        print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 for manipulator.')
    parser.add_argument('--code_version', type=str, default='td3_31')
    parser.add_argument('--vis_port', type=int, default=6016)

    parser.add_argument('--max_episode_steps', type=int, default=100)
    parser.add_argument('--distance_threshold', type=float, default=0.02)
    parser.add_argument('--reward_type', type=str, default='dense')
    parser.add_argument('--max_angles_vel', type=float, default=10.)
    parser.add_argument('--num_joints', type=int, default=12)
    parser.add_argument('--num_segments', type=int, default=2)
    parser.add_argument('--plane_model', type=bool, default=True)
    parser.add_argument('--cc_model', type=bool, default=False)
    parser.add_argument('--goal_set', type=str, choices=['easy', 'hard', 'super hard', 'random'],
                        default='random')
    parser.add_argument('--collision_cnt', type=int, default=13)
    parser.add_argument('--scene_file', type=str, default='simple_12_1.ttt')
    parser.add_argument('--headless_mode', type=bool, default=False)

    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--episodes', type=int, default=10000)

    args = parser.parse_args()
    # write the selected car to configuration file

    playGame(args, args.train, args.episodes)




