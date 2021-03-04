import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'PPO'))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'vrep_pyrep'))
from ppo_agent import PPO_agent, ReplayBuffer
from actor_critic import Actor_critic
import argparse
import torch
import matplotlib.pyplot as plt
from vrep_pyrep.env import ManipulatorEnv
import random
import numpy as np
import time


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


def main(args_):
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

    obs_dims = env.observation_space.shape[0]
    act_dims = env.action_space.shape[0]
    print(f'obs_dims is {obs_dims}')
    print(f'act_dims is {act_dims}')

    pooling = ReplayBuffer(buffer_size=args_.buffer_size,
                           act_dims=act_dims, obs_dims=obs_dims,
                           batch_size=args_.batch_size)
    actor_critic = Actor_critic(env=env,
                                actor_obs_dims=obs_dims,
                                actor_hidden_sizes=args_.actor_hidden_dim,
                                actor_action_dims=act_dims,
                                critic_obs_dims=obs_dims,
                                critic_hidden_sizes=args_.critic_hidden_dim)

    ppo = PPO_agent(args=args_,
                    env=env,
                    actor_critic=actor_critic,
                    num_episodes=args_.num_episodes,
                    max_steps_per_episodes=args_.max_episode_steps,
                    pooling=pooling,
                    clip_epsilon=args_.clip_epsilon,
                    gamma=args_.gamma,
                    lr=args_.lr,
                    ppo_epoch=args_.ppo_epoch,
                    weight_epsilon=args_.ent_coef)
    if args_.train:
        ppo.train()
        env.end_simulation()
        time.sleep(2)
    else:
        # ppo.eval_model(f'/home/cq/code/manipulator/PPO/checkpoints/mani_34/994.pth', 10)
        path_len = 0
        rewards = 0
        result = 0
        model = torch.load(f'/home/cq/code/manipulator/PPO/checkpoints/mani_37/997.pth')
        actor_critic.load_state_dict(model)
        for i in range(10):
            # model = torch.load(f'/home/cq/code/manipulator/PPO/checkpoints/mani_34/.pth')
            # actor_critic.load_state_dict(model)
            cur_obs = env.reset()
            for i in range(args_.max_episode_steps):
                action = actor_critic.eval_action(torch.FloatTensor(cur_obs[None]))
                next_obs, reward, done, info = env.step(action)
                rewards += reward
                path_len += 1
                cur_obs = next_obs
                if done:
                    eval_result = 0.
                    if done and path_len < args_.max_episode_steps and not any(info['collision_state']):
                        eval_result = 1.
                    result += eval_result
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-version', type=str, default='mani_36')
    parser.add_argument('--visdom-port', type=int, default=6016)
    parser.add_argument('--seed', type=int, default=1)
    # ppo config
    parser.add_argument('--actor-hidden-dim', type=list, default=[128, 128])
    parser.add_argument('--critic-hidden-dim', type=list, default=[64, 64])
    parser.add_argument('--eval-times', type=int, default=1)
    parser.add_argument('--eval-freq', type=int, default=5)
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--clip-epsilon', type=int, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--ent-coef', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=3e-4)
    # buffer config
    parser.add_argument('--buffer-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    # env config
    parser.add_argument('--max-episode-steps', type=int, default=100)
    parser.add_argument('--distance-threshold', type=float, default=0.02)
    parser.add_argument('--reward-type', type=str, default='dense distance')
    parser.add_argument('--max-angles-vel', type=float, default=10.)
    parser.add_argument('--num-joints', type=int, default=12)
    parser.add_argument('--num-segments', type=int, default=2)
    parser.add_argument('--plane-model', action='store_true')
    parser.add_argument('--cc-model', action='store_true')
    parser.add_argument('--goal-set', type=str, choices=['easy', 'hard', 'super hard', 'random'],
                        default='hard')
    parser.add_argument('--collision-cnt', type=int, default=15)
    parser.add_argument('--scene-file', type=str, default='simple_12_1.ttt')
    parser.add_argument('--headless-mode', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    logger = get_logger(args.code_version)
    main(args)
