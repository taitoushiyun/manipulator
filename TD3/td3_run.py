import importmagic
from vrep_con.vrep_utils import ManipulatorEnv
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

def playGame(args_, train=True, episode_count=2000):
    # goal_index = {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20],
    #               'hard': [0, 20, 0, 15, 0, 20, 0, 20, 0, 20],
    #               'super hard': [0, -50, 0, -50, 0, -50, 0, 0, -20, -10]}
    # env_config = {
    #     'distance_threshold': args_.distance_threshold,
    #     'reward_type': args_.reward_type,
    #     'max_angles_vel': args_.max_angles_vel,  # 10degree/s
    #     'num_joints': args_.num_joints,
    #     'goal_set': goal_index[args_.goal_set],
    # }
    # env = ManipulatorEnv(0, env_config)
    env = gym.make('LunarLanderContinuous-v2')

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

        print("TORCS Experiment Start.")
        best = -np.inf

        if train:
            vis = visdom.Visdom(port=6016, env='td3_1')
            td3_torcs(env, agent, episode_count, 1000, 'checkpoints', vis)
        else:
            state = env.reset()
            total_reward = 0
            for t in count():
                action = agent.act(state, add_noise=False)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    print(f"Total reward: {total_reward}")
                    print(f"Episode length: {t}")
                    break

    finally:
        env.end_simulation()  # This is for shutting down TORCS
        print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 for manipulator.')
    parser.add_argument('--distance-threshold', type=float, default=0.02)
    parser.add_argument('--reward-type', type=str, default='dense')
    parser.add_argument('--max-angles-vel', type=float, default=10.)
    parser.add_argument('--num-joints', type=int, default=10)
    parser.add_argument('--goal-set', type=str, choices=['easy', 'hard', 'super hard'], default='hard')

    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--episodes', type=int, default=2000)

    args = parser.parse_args()
    # write the selected car to configuration file
    playGame(args, args.train, args.episodes)

