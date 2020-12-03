import os
import numpy as np
import torch
import argparse
from PPO_multi.logger_utility import logger
from PPO_multi.utils import set_global_seed
from PPO_multi.learner import PPOLearner
from PPO_multi.actor import SamplerWorker
from PPO_multi.model import ActorCritic
from PPO_multi.monitor import plot
from PPO_multi.utils import save_model
import multiprocessing as mp
from vrep_con.vrep_utils import ManipulatorEnv

import gym



def run_worker(index, worker_config, remote, shared_policy, reward_record, path_len_record):
    # env = ManipulatorEnv(index, num_joints=10)
    env =  gym.make('CartPole-v0')
    policy = ActorCritic(obs_dim=worker_config['obs_dim'],
                         actor_hidden=worker_config['actor_hidden'],
                         critic_hidden=worker_config['critic_hidden'],
                         action_dim=worker_config['act_dim'],
                         network=None)
    sampler = SamplerWorker(env=env,
                            policy=policy,
                            gamma=worker_config['gamma'],
                            lammbda=worker_config['lammbda'],
                            batch_size=worker_config['batch_size'],
                            act_dim=worker_config['act_dim'],
                            obs_dim=worker_config['obs_dim'],
                            reward_record=reward_record,
                            reward_record_size=worker_config['reward_record_size'],
                            path_len_record=path_len_record,
                            path_len_record_size=worker_config['path_len_record_size']
                            )
    while True:
        cmd = remote.recv()
        if cmd == 'step':
            sampler.policy.load_state_dict(shared_policy.state_dict())
            sample_data = sampler.run()
            remote.send(sample_data)
        elif cmd == 'close':
            remote.close()
            break


def main(args):
    main_dir = os.path.abspath(os.path.dirname(__file__))
    checkpoints_dir = os.path.join(main_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    parser.add_argument('--checkpoints-dir', type=str, default=checkpoints_dir)
    args = parser.parse_args()
    param = ''
    for key, item in vars(args).items():
        param += str(key) + ': ' + str(item) + '\n'
    logger.info(param)

    set_global_seed(args.seed)

    shared_policy = ActorCritic(obs_dim=args.obs_dim,
                         actor_hidden=args.actor_hidden,
                         critic_hidden=args.critic_hidden,
                         action_dim=args.act_dim,
                         network=None)
    ppo_learner = PPOLearner(policy=shared_policy,
                             batch_size=args.batch_size,
                             minibatch_size=args.minibatch_size,
                             cliprange=args.clip_range,
                             gamma=args.gamma,
                             lr=args.lr,
                             epochs=4,
                             max_grad_norm=args.max_grad_norm,
                             entropy_coef=args.ent_coef,
                             vf_coef=0.5)
    mp_manager = mp.Manager()
    iter_step = mp.Value('i', 0)
    reward = mp_manager.list()
    path_len = mp_manager.list()
    worker_config = mp_manager.dict({'num_joints': args.num_joints,
                                     'obs_dim': args.obs_dim,
                                     'act_dim': args.act_dim,
                                     'actor_hidden': args.actor_hidden,
                                     'critic_hidden': args.critic_hidden,
                                     'batch_size': args.batch_size,
                                     'gamma': args.gamma,
                                     'lammbda': args.lammbda,
                                     'reward_record_size': args.reward_record_size,
                                     'path_len_record_size': args.path_len_record_size
                                     })
    remotes, remote_workers = zip(*[mp.Pipe() for _ in range(args.num_envs)])
    processes =  []
    for i in range(args.num_envs):
        processes.append(mp.Process(target=run_worker, args=(i, worker_config, remote_workers[i], shared_policy,
                                                             reward, path_len)))
    for remote_worker in remote_workers:
        remote_worker.close()
    processes.append(mp.Process(target=plot, args=(args, iter_step, reward, path_len)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    assert args.batch_size % args.minibatch_size == 0
    for update_step in range(args.total_update):
        for remote in remotes:
            remote.send('step')
        sample_data = [remote.recv() for remote in remotes]
        returns, values, old_log_probs, observations, actions = map(np.asarray, zip(*sample_data))
        ppo_learner.learn(observations=torch.tensor(observations, dtype=torch.float, device=args.device),
                          actions=torch.tensor(actions, dtype=torch.float, device=args.device),
                          values=torch.tensor(values, dtype=torch.float, device=args.device),
                          returns=torch.tensor(returns, dtype=torch.float, device=args.device),
                          old_log_probs=torch.tensor(old_log_probs, dtype=torch.float, device=args.device))
        iter_step += 1


        if update_step % args.save_interval:
            save_file_name = os.path.join(checkpoints_dir, '%.5i' % update_step + '.pt')
            save_model(shared_policy, save_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ppo_multi')
    parser.add_argument('--network', type=str, default='mlp',          # TODO: change here if use mlp to test
                        choices=['cnn', 'lstm', 'mlp', 'impala_cnn', 'gfootball_impala_cnn'])
    parser.add_argument('--seed', type=int, default=19940208)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--num-joints', type=int, default=10)
    parser.add_argument('--obs-dim', type=int, default=22)
    parser.add_argument('--act-dim', type=int, default=5)
    parser.add_argument('--actor-hidden', type=list,  default=[30, 30])
    parser.add_argument('--critic-hidden', type=list, default=[30, 30])

    parser.add_argument('--total-update', type=int, default=1e+4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--minibatch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lammbda', type=float, default=0.95)
    parser.add_argument('--ent-coef', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=5)

    parser.add_argument('--reward-record-size', type=int, default=10)
    parser.add_argument('--path-len-record-size', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--plot-interval', type=int, default=30)
    parser.add_argument('--vis-port', type=int, default=6016)
    parser.add_argument('--code-version', type=str, default='mani_ppo_multi_0')
    args = parser.parse_args()
    main(args)
