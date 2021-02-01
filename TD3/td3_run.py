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
    }
    env = ManipulatorEnv(env_config)
    env.action_space.seed(args_.seed)
    # env = gym.make('LunarLanderContinuous-v2')

    agent = TD3Agent(state_size=env.observation_space.shape[0],
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
            from _collections import deque
            result_queue = deque(maxlen=20)
            for i in range(100):
                # if i % 5 == 0:
                model = torch.load(
                    f'/home/cq/code/manipulator/TD3/checkpoints/td3_49/999.pth')  # 'PPO/checkpoints/40.pth'
                    # f'/media/cq/系统/Users/Administrator/Desktop/实验记录/td3_14/checkpoints/actor/1000.pth')
                agent.actor_local.load_state_dict(model)

                state = env.reset()
                total_reward = 0
                path_length = 0
                for t in count():
                    action = agent.act(state, add_noise=False)
                    next_state, reward, done, info = env.step(action)
                    time.sleep(0.1)
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
    parser.add_argument('--code-version', type=str, default='td3_52')
    parser.add_argument('--vis-port', type=int, default=6016)
    parser.add_argument('--seed', type=int, default=0)
    #  TD3 config
    parser.add_argument('--actor-hidden', type=list, default=[100, 100])
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
    # env config
    parser.add_argument('--max-episode-steps', type=int, default=100)
    parser.add_argument('--distance-threshold', type=float, default=0.02)
    parser.add_argument('--reward-type', type=str, default='dense potential')
    parser.add_argument('--max-angles-vel', type=float, default=10.)
    parser.add_argument('--num-joints', type=int, default=12)
    parser.add_argument('--num-segments', type=int, default=2)
    parser.add_argument('--plane-model', type=bool, default=False)
    parser.add_argument('--cc-model', type=bool, default=False)
    parser.add_argument('--goal-set', type=str, choices=['easy', 'hard', 'super hard', 'random'],
                        default='hard')
    parser.add_argument('--collision-cnt', type=int, default=15)
    parser.add_argument('--scene-file', type=str, default='simple_12_1.ttt')
    parser.add_argument('--headless-mode', type=bool, default=False)

    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=1000)

    args = parser.parse_args()
    # write the selected car to configuration file
    logger = get_logger(args.code_version)
    playGame(args, args.train, args.episodes)




