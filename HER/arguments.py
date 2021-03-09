import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--code-version', type=str, default='her_20')
    parser.add_argument('--vis-port', type=int, default=6016)
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=1000, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=10, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio') #TODO
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')  # TODO
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')  # TODO
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')  # TODO
    parser.add_argument('--demo-length', type=int, default=18, help='the demo length')
    parser.add_argument('--demo-dense', type=int, default=18)
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')


    # env config
    parser.add_argument('--max-episode-steps', type=int, default=50)
    parser.add_argument('--distance-threshold', type=float, default=0.01)
    parser.add_argument('--reward-type', type=str, default='sparse')
    parser.add_argument('--max-angles-vel', type=float, default=10.)
    parser.add_argument('--num-joints', type=int, default=24)
    parser.add_argument('--num-segments', type=int, default=2)
    parser.add_argument('--plane-model', action='store_true')
    parser.add_argument('--cc-model', action='store_true')
    parser.add_argument('--goal-set', type=str, choices=['easy', 'hard', 'super hard', 'random',
                                                         'block0', 'block1', 'block2', 'block3', 'block4',
                                                         'draw0'],
                        default='random')
    parser.add_argument('--collision-cnt', type=int, default=15)
    parser.add_argument('--scene-file', type=str, default='mani_env_24.xml')
    parser.add_argument('--headless-mode', action='store_true')
    parser.add_argument('--random-initial-state', action='store_true')

    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    return args