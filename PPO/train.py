from PPO.ppo_agent import PPO_agent, ReplayBuffer
from PPO.actor_critic import Actor_critic
import argparse
import torch
import matplotlib.pyplot as plt
from vrep_con.vrep_utils import ManipulatorEnv


def main(args_):
    env_config = {
        'distance_threshold': args_.distance_threshold,
        'reward_type': args_.reward_type,
        'max_angles_vel': args_.max_angles_vel,  # 10degree/s
        'num_joints': args.num_joints,
    }
    env = ManipulatorEnv(0, env_config)
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
    ppo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-version', type=str, default='main_22')
    parser.add_argument('--visdom-port', type=int, default=6016)
    parser.add_argument('--actor-hidden-dim', type=list, default=[64, 64])
    parser.add_argument('--critic-hidden-dim', type=list, default=[64, 64])
    # ppo config
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--max-episode-steps', type=int, default=100)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--clip-epsilon', type=int, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--ent-coef', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=3e-4)
    # buffer config
    parser.add_argument('--buffer-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    # env config
    parser.add_argument('--distance-threshold', type=float, default=0.02)
    parser.add_argument('--reward-type', type=str, default='dense')
    parser.add_argument('--max-angles-vel', type=float, default=10.)
    parser.add_argument('--num-joints', type=int, default=10)
    args = parser.parse_args()
    main(args)
