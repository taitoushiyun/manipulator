import argparse
from vrep_con.vrep_utils import ManipulatorEnv
from TD3.policy import Policy, QFun
from TD3.sampler import ReplayBuffer, Sampler
from TD3.td3 import TD3


def main(args_):
    goal_index = {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20],
                  'hard': [0, 20, 0, 15, 0, 20, 0, 20, 0, 20],
                  'super hard': [0, -50, 0, -50, 0, -50, 0, 0, -20, -10]}
    env_config = {
        'distance_threshold': args_.distance_threshold,
        'reward_type': args_.reward_type,
        'max_angles_vel': args_.max_angles_vel,  # 10degree/s
        'num_joints': args_.num_joints,
        'goal_set': goal_index[args_.goal_set],
    }
    env = ManipulatorEnv(0, env_config)
    obs_dims = env.observation_space.shape[0]
    act_dims = env.action_space.shape[0]
    policy = Policy(obs_dim=obs_dims,
                    act_dim=act_dims,
                    max_action=env_config['max_angles_vel'],
                    max_sigma=1,
                    min_sigma=0.05,
                    decay_period=500)
    target_policy = Policy(obs_dim=obs_dims,
                           act_dim=act_dims,
                           max_action=env_config['max_angles_vel'],
                           max_sigma=1,
                           min_sigma=0.05,
                           decay_period=500)
    target_policy.load_state_dict(policy.state_dict())
    qf1 = QFun(obs_dim=obs_dims,
               act_dim=act_dims)
    qf2 = QFun(obs_dim=obs_dims,
               act_dim=act_dims)
    target_qf1 = QFun(obs_dim=obs_dims,
                      act_dim=act_dims)
    target_qf2 = QFun(obs_dim=obs_dims,
                      act_dim=act_dims)
    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())

    pool = ReplayBuffer(max_memory_size=args.max_buffer_size,
                        obs_dims=obs_dims,
                        act_dims=act_dims)
    sampler = Sampler(env, policy, pool,
                      min_pool_size=args.min_pool_size,
                      max_episode_steps=args.max_episode_steps)
    td3 = TD3(env=env,
              policy=policy,
              target_policy=target_policy,
              pool=pool,
              sampler=sampler,
              qf1=qf1,
              qf2=qf2,
              target_qf1=target_qf1,
              target_qf2=target_qf2,
              target_policy_noise=args.target_policy_noise,
              target_policy_noise_clip=args.target_policy_noise_clip,
              tau=args.tau,
              gamma=args.gamma,
              update_freq=args.update_freq,
              num_epoches=args.num_epoches,
              epoch_len=args.epoch_len,
              save_freq=args.save_freq,
              code_version=args.code_version,
              vis_port=args.vis_port
              )
    td3.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # env config
    parser.add_argument('--distance-threshold', type=float, default=0.02)
    parser.add_argument('--reward-type', type=str, default='dense')
    parser.add_argument('--max-angles-vel', type=float, default=10.)
    parser.add_argument('--num-joints', type=int, default=10)
    parser.add_argument('--goal-set', type=str, choices=['easy', 'hard', 'super hard'], default='hard')
    # pool config
    parser.add_argument('--max_buffer_size', type=int, default=1000000)
    parser.add_argument('--min_pool_size', type=int, default=10000)
    parser.add_argument('--max_episode_steps', type=int, default=100)

    parser.add_argument('--target_policy_noise', type=float, default=0.2)
    parser.add_argument('--target_policy_noise_clip', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--update_freq', type=int, default=500)
    parser.add_argument('--num_epoches', type=int, default=10000)
    parser.add_argument('--epoch_len', type=int , default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--code_version', type=int, default='td3_0')
    parser.add_argument('--vis_port', type=int, default=6016)





    args = parser.parse_args()
    main(args)
