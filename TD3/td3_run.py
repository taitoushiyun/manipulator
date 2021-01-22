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
            td3_torcs(env, agent, episode_count, args.max_episode_steps, 'checkpoints', vis)
        else:
            for i in range(4900, 5000):
                if i % 5 == 0:
                    model = torch.load(f'I://remote/manipulator/TD3/checkpoints/actor/{i}.pth')  # 'PPO/checkpoints/40.pth'
                    agent.actor_local.load_state_dict(model)
                    state = env.reset(eval_=True)
                    total_reward = 0
                    path_length = 0
                    for t in count():
                        action = agent.act(state, add_noise=False)
                        next_state, reward, done, _ = env.step(action)
                        total_reward += reward
                        path_length += 1
                        state = next_state
                        if done or path_length >= args.max_episode_steps:
                            print(f'episode {i}')
                            print(f"Total reward: {total_reward}")
                            print(f"Episode length: {t+1}")
                            time.sleep(1)
                            break

    finally:
        env.end_simulation()  # This is for shutting down TORCS
        print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 for manipulator.')
    parser.add_argument('--code_version', type=str, default='td3_25')
    parser.add_argument('--vis_port', type=int, default=6016)

    parser.add_argument('--max_episode_steps', type=int, default=100)
    parser.add_argument('--distance_threshold', type=float, default=0.02)
    parser.add_argument('--reward_type', type=str, default='dense')
    parser.add_argument('--max_angles_vel', type=float, default=10.)
    parser.add_argument('--num_joints', type=int, default=12)
    parser.add_argument('--num_segments', type=int, default=2)
    parser.add_argument('--plane_model', type=bool, default=True)
    parser.add_argument('--cc_model', type=bool, default=True)
    parser.add_argument('--goal_set', type=str, choices=['easy', 'hard', 'super hard', 'random', ''],
                        default='random')
    parser.add_argument('--collision_cnt', type=int, default=15)
    parser.add_argument('--scene_file', type=str, default='by_12_1.ttt')
    parser.add_argument('--headless_mode', type=bool, default=True)

    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--episodes', type=int, default=5000)

    args = parser.parse_args()
    # write the selected car to configuration file
    try:
        playGame(args, args.train, args.episodes)
    except:
        raise EOFError



