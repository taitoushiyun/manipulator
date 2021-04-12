import torch
import time
import os
from datetime import datetime
import numpy as np
# from mpi4py import MPI
# from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import Actor, Critic, ActorDense, CriticDense, ActorDenseSimple, CriticDenseSimple, ActorDenseASF
from rl_modules.models import DNet, PopArtCritic
from rl_modules.pop_art import PopArt
from mpi_utils.normalizer import normalizer, normalizer_torch
from her_modules.her import her_sampler
import visdom
from _collections import deque
import logging
logger = logging.getLogger('mani')


DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi


def normalize(y, pop_art):
    if pop_art is None:
        return y
    return (y - pop_art.mu) / pop_art.sigma


def denormalize(y, pop_art):
    if pop_art is None:
        return y
    return pop_art.sigma * y + pop_art.mu


class HeatBuffer:
    def __init__(self, buffer_size=2000):
        self.heat_map = np.zeros((201, 121))
        self.buffer_size = buffer_size
        self.achieved_goal = np.zeros((self.buffer_size, 3))
        self._top = 0
        self._size = 0

    def add_sample(self, obs):
        assert obs.shape[0] == 3
        old_obs = self.achieved_goal[self._top]
        old_index = self._get_index(old_obs)
        new_index = self._get_index(obs)
        self.heat_map[old_index] = max(self.heat_map[old_index] - 1, 0)
        self.heat_map[new_index] = self.heat_map[new_index] + 1
        self.achieved_goal[self._top] = obs.copy()
        self._advance()

    def _get_index(self, obs):
        index_x = self.clamp((obs[0] - 0.2) // 0.01, 0, 120)
        index_y = self.clamp(obs[2] // 0.01, 0, 200)
        return int(index_y), int(index_x)

    def _advance(self):
        self._top = (self._top + 1) % self.buffer_size
        if self._size < self.buffer_size:
            self._size += 1

    @staticmethod
    def clamp(n, minn, maxn):
        return max(min(maxn,  n), minn)

    @property
    def total_size(self):
        return self._size


class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        if self.args.actor_type == 'normal':
            actor = Actor
        elif self.args.actor_type == 'dense':
            actor = ActorDense
        elif self.args.actor_type == 'dense_simple':
            actor = ActorDenseSimple
        elif self.args.actor_type == 'dense_asf':
            actor = ActorDenseASF
        else:
            raise ValueError
        if self.args.critic_type == 'normal':
            critic = Critic
        elif self.args.critic_type == 'dense':
            critic = CriticDense
        elif self.args.critic_type == 'dense_simple':
            critic = CriticDenseSimple
        elif self.args.critic_type == 'pop_art':
            critic = PopArtCritic
        else:
            raise ValueError
        self.env = env
        self.env_params = env_params
        self.actor_network = actor(env_params, args)
        self.critic_network = critic(env_params, args)
        self.critic_explore_network = critic(env_params, args)
        self.predict_network = DNet(env_params, args)

        # build up the target network
        self.actor_target_network = actor(env_params, args)
        self.critic_target_network = critic(env_params, args)
        self.critic_explore_target_network = critic(env_params, args)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_explore_target_network.load_state_dict(self.critic_explore_network.state_dict())

        if self.args.critic_type == 'pop_art':
            self.pop_art = PopArt(args)
        else:
            self.pop_art = None
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.critic_explore_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            self.critic_explore_target_network.cuda()
            self.predict_network.cuda()
            self.pop_art.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.lower_layer.parameters(), lr=self.args.lr_critic)
        self.critic_explore_optim = torch.optim.Adam(self.critic_explore_network.parameters(), lr=self.args.lr_critic)
        self.predict_optim = torch.optim.Adam(self.predict_network.parameters(), lr=self.args.lr_predict)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        device = 'cuda' if self.args.cuda else 'cpu'
        self.q_norm = normalizer_torch(device=device, size=1, default_clip_range=self.args.clip_range)
        self.q_explore_norm = normalizer_torch(device=device, size=1, default_clip_range=self.args.clip_range)
        self.action_l2_norm = normalizer_torch(device=device, size=env_params['action'], default_clip_range=self.args.clip_range)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.code_version)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.vis = visdom.Visdom(port=args.vis_port, env=args.code_version)
        self.vis.line(X=[0], Y=[0], win='result', opts=dict(Xlabel='episode', Ylabel='result', title='result'))
        self.vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
        self.vis.line(X=[0], Y=[0], win='success rate', opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='success rate'))
        self.vis.line(X=[0], Y=[0], win='eval result', opts=dict(Xlabel='episode', Ylabel='eval result', title='eval result'))
        self.vis.line(X=[0], Y=[0], win='eval path len', opts=dict(Xlabel='episode', Ylabel='len', title='eval path len'))
        self.vis.line(X=[0], Y=[0], win='eval success rate', opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='eval success rate'))
        if args.goal_set != 'random':
            self.vis.line(X=[0], Y=[0], win='reward', opts=dict(Xlabel='episode', Ylabel='reward', title='reward'))
            self.vis.line(X=[0], Y=[0], win='mean reward', opts=dict(Xlabel='episode', Ylabel='reward', title='mean reward'))
            self.vis.line(X=[0], Y=[0], win='eval reward', opts=dict(Xlabel='episode', Ylabel='reward', title='eval reward'))
        # self.debug_items = {'actor_reward_loss': {'xlabel': 'episode', 'ylabel': 'loss', 'title': 'actor_reward_loss'},
        #                     'actor_action_l2_loss': {'xlabel': 'episode', 'ylabel': 'loss', 'title': 'actor_action_l2_loss'},
        #                     'actor_action_loss': {'xlabel': 'episode', 'ylabel': 'loss', 'title': 'actor_action_loss'},
        #                     'actor_loss': {'xlabel': 'episode', 'ylabel': 'loss', 'title': 'actor_loss'},
        #                     'critic_action_loss': {'xlabel': 'episode', 'ylabel': 'loss', 'title': 'critic_action_loss'},
        #                     'critic_reward_loss': {'xlabel': 'episode', 'ylabel': 'loss', 'title': 'critic_reward_loss'},
        #                     'critic_loss': {'xlabel': 'episode', 'ylabel': 'loss', 'title': 'critic_loss'},
        #                     'q_value': {'xlabel': 'episode', 'ylabel': 'value', 'title': 'q_value',
        #                                  'legend': ['target_q', 'real_q']},
        #                     'pop_art': {'xlabel': 'episode', 'ylabel': 'value', 'title': 'pop_art',
        #                                 'legend': ['mu', 'mu+sigma', 'mu-sigma']},
        #                     'pop_is_active': {'xlabel': 'episode', 'ylabel': 'value', 'title': 'pop_is_active'},
        #                     'r_tensor': {'xlabel': 'episode', 'ylabel': 'value', 'title': 'r_tensor'},
        #                     }
        self.vis.line(X=[0], Y=[0], win='action_critic_loss', opts=dict(Xlabel='episode', Ylabel='loss', title='action_critic_loss'))
        self.vis.line(X=[0], Y=[0], win='action_l2_loss', opts=dict(Xlabel='episode', Ylabel='loss', title='action_l2_loss'))
        self.vis.line(X=[0], Y=[0], win='q_action_loss', opts=dict(Xlabel='episode', Ylabel='loss', title='q_action_loss'))
        self.vis.line(X=[0], Y=[0], win='actor_loss', opts=dict(Xlabel='episode', Ylabel='loss', title='actor_loss'))
        self.vis.line(X=[0], Y=[0], win='actor_loss_1', opts=dict(Xlabel='episode', Ylabel='loss', title='actor_loss_1'))
        self.vis.line(X=[0], Y=[0], win='critic_loss', opts=dict(Xlabel='episode', Ylabel='loss', title='critic_loss'))
        self.vis.line(X=[0], Y=[0], win='pop art', opts=dict(Xlabel='episode', Ylabel='mu', title='pop art',
                                                             legend=['mu', 'mu+sigma', 'mu-sigma']))
        self.vis.line(X=[0], Y=[0], win='target_q', opts=dict(Xlabel='episode', Ylabel='target_q', title='target_q',
                                                              legend=['target_q', 'real_q']))
        self.vis.line(X=[0], Y=[0], win='pop_is_active', opts=dict(Xlabel='episode', Ylabel='active_state', title='pop_is_active'))
        self.vis.line(X=[0], Y=[0], win='r_tensor',
                      opts=dict(Xlabel='episode', Ylabel='value', title='r_tensor'))
        self.vis.line(X=[0], Y=[0], win='cnt_enough',
                      opts=dict(Xlabel='episode', Ylabel='value', title='cnt_enough'))
        self.result_deque = deque(maxlen=20)
        self.score_deque = deque(maxlen=10)
        self.eval_result_queue = deque(maxlen=10)
        self.heat_buffer = HeatBuffer(buffer_size=2000)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        i_episode = 0
        for epoch in range(self.args.n_epochs):
            for n_cycle in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                time_a = time.time()
                for _ in range(self.args.num_rollouts_per_mpi):

                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset(i_epoch=epoch)
                    # last_achieved_goal = observation['achieved_goal']
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    episode_length = 0
                    score = 0
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        # self.env.render()
                        if not self.args.headless_mode:
                            self.env.render()
                        observation_new, reward, done, info = self.env.step(action)
                        # if np.linalg.norm(observation_new['achieved_goal'] - last_achieved_goal) > 0.005:
                        #     self.heat_buffer.add_sample(observation_new['achieved_goal'])
                        # last_achieved_goal = observation_new['achieved_goal'].copy()
                        episode_length += 1
                        score += reward
                        if episode_length >= self.args.max_episode_steps:
                            if info['is_success']:
                                result = 1.
                            else:
                                result = 0
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)

                    i_episode += 1
                    self.result_deque.append(result)
                    self.score_deque.append(score)
                    success_rate = np.mean(self.result_deque)
                    mean_score = np.mean(self.score_deque)

                    logger.info("Episode: %d,          Path length: %d       result: %f       reward: %f"
                                % (i_episode, episode_length, result, score))
                    if self.args.goal_set != 'random':
                        if self.args.reward_type == 'dense potential':
                            self.vis.line(X=[i_episode], Y=[(score - self.env.max_rewards) * 100], win='reward', update='append')
                            self.vis.line(X=[i_episode], Y=[(mean_score - self.env.max_rewards) * 100], win='mean reward', update='append')
                        if self.args.reward_type == 'dense distance':
                            self.vis.line(X=[i_episode], Y=[score], win='reward', update='append')
                            self.vis.line(X=[i_episode], Y=[mean_score], win='mean reward', update='append')
                    self.vis.line(X=[i_episode], Y=[result], win='result', update='append')
                    self.vis.line(X=[i_episode], Y=[episode_length], win='path len', update='append')
                    self.vis.line(X=[i_episode], Y=[success_rate * 100], win='success rate', update='append')

                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                action_critic_loss_batch = 0
                action_l2_loss_batch = 0
                q_action_loss_batch = 0
                actor_loss_batch = 0
                actor_loss1_batch = 0
                critic_loss_batch = 0
                mu_batch = 0
                sigma_batch = 0
                target_q_batch = 0
                real_q_batch = 0
                r_tensor_batch = 0
                for _ in range(self.args.n_batches):
                    # train the network
                    action_critic_loss, actor_loss1, action_l2_loss, q_action_loss, actor_loss, critic_loss, \
                        mu, sigma, target_q, real_q, r_tensor = self._update_network()
                    action_critic_loss_batch += action_critic_loss
                    action_l2_loss_batch += action_l2_loss
                    q_action_loss_batch += q_action_loss
                    actor_loss_batch += actor_loss
                    actor_loss1_batch += actor_loss1
                    critic_loss_batch += critic_loss
                    mu_batch += mu
                    sigma_batch += sigma
                    target_q_batch += target_q
                    real_q_batch += real_q
                    r_tensor_batch += r_tensor
                self.vis.line(X=[i_episode], Y=[action_critic_loss_batch], win='action_critic_loss', update='append')
                self.vis.line(X=[i_episode], Y=[action_l2_loss_batch], win='action_l2_loss', update='append')
                self.vis.line(X=[i_episode], Y=[q_action_loss_batch], win='q_action_loss', update='append')
                self.vis.line(X=[i_episode], Y=[actor_loss_batch], win='actor_loss', update='append')
                self.vis.line(X=[i_episode], Y=[actor_loss1_batch], win='actor_loss_1', update='append')
                self.vis.line(X=[i_episode], Y=[critic_loss_batch], win='critic_loss', update='append')
                self.vis.line(X=[i_episode], Y=[mu_batch / self.args.n_batches], win='pop art', update='append', name='mu')
                self.vis.line(X=[i_episode], Y=[(mu_batch+sigma_batch) / self.args.n_batches], win='pop art', update='append', name='mu+sigma')
                self.vis.line(X=[i_episode], Y=[(mu_batch-sigma_batch) / self.args.n_batches], win='pop art', update='append', name='mu-sigma')
                self.vis.line(X=[i_episode], Y=[target_q_batch / self.args.n_batches], win='target_q', update='append', name='target_q')
                self.vis.line(X=[i_episode], Y=[real_q_batch / self.args.n_batches], win='target_q', update='append', name='real_q')
                self.vis.line(X=[i_episode], Y=[self.pop_art.pop_is_active], win='pop_is_active', update='append')
                self.vis.line(X=[i_episode], Y=[self.pop_art.cnt_enough], win='cnt_enough', update='append')
                self.vis.line(X=[i_episode], Y=[r_tensor_batch / self.args.n_batches], win='r_tensor', update='append')
                # soft update
                # print(target_q_batch / self.args.n_batches, real_q_batch / self.args.n_batches)
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network.lower_layer, self.critic_network.lower_layer)
                # if self.args.double_q:
                #     self._soft_update_target_network(self.critic2_target_network, self.critic2_network)
                time_b = time.time()
                print(time_b - time_a)
            # start to do the evaluation
            if epoch > 0.9 * self.args.n_epochs:
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.actor_network.state_dict()], self.model_path + f'/{epoch}.pt')
            else:
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.actor_network.state_dict()], self.model_path + f'/model.pt')
            self._eval_agent(epoch)
            # if epoch % 2 == 0:
            #     self.vis.heatmap(
            #         X=self.heat_buffer.heat_map,
            #         win=f'epoch{epoch}',
            #         opts={
            #             'Xlable': 'X',
            #             'Ylable': 'Y',
            #             'title': f'epoch{epoch}',
            #             'columnnames': list(map(lambda x: '%.2f' % x, list(np.linspace(0.2, 1.4, num=121, endpoint=True)))),
            #             'rownames': list(map(lambda x: '%.2f' % x, list(np.linspace(0, 2, num=201, endpoint=True)))),
            #             'colormap': 'Viridis',  # 'Electric'
            #         }
            #     )
            print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        # print((r_tensor+1).sum())
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # calculate the target Q value function
        with torch.no_grad():
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next).detach()
            q_next_value = denormalize(q_next_value, self.pop_art)
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            r_tensor_out = r_tensor
            target_q_value_output = target_q_value
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

            obs_predicted_target = inputs_next_norm_tensor[..., :-3].detach()
        # print((target_q_value+1).sum())
        # predict loss
        obs_predicted = self.predict_network.forward(inputs_norm_tensor, actions_tensor)
        predict_loss_ = (obs_predicted - obs_predicted_target).pow(2)
        predict_loss = predict_loss_.mean()
        self.predict_optim.zero_grad()
        predict_loss.backward()
        self.predict_optim.step()

        # q explore loss
        real_q_explore_value = self.critic_explore_network(inputs_norm_tensor, actions_tensor)
        target_q_explore_value = predict_loss_.mean(dim=-1, keepdim=True).detach()
        # self.q_explore_norm.update(target_q_explore_value)
        # target_q_explore_value = self.q_explore_norm.normalize(target_q_explore_value)
        critic_explore_loss = (target_q_explore_value.detach() - real_q_explore_value).pow(2).mean()


        # the q loss
        if self.args.critic_type == "pop_art":
            self.pop_art.update(target_q_value, [self.critic_network, self.critic_target_network])
            target_q_value = normalize(target_q_value, self.pop_art)
        else:
            self.critic_network.art(target_q_value.mean())
            self.critic_network.update_mu()
        # print(self.critic_network.sigma)
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        # self.q_norm.update(target_q_value)
        # target_q_value = self.q_norm.normalize(target_q_value)
        critic_reward_loss = (target_q_value - real_q_value).pow(2).mean()
        critic_loss = (1 - self.args.explore_ratio) * critic_reward_loss + self.args.explore_ratio * critic_explore_loss

        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss_1 = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        # self.action_l2_norm.update(actions_real.detach())
        # action_real_norm = self.action_l2_norm.normalize(actions_real)
        actor_loss_2 = self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        actor_loss = actor_loss_1 + actor_loss_2


        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return (0,
                actor_loss_1.detach().item(),
                actor_loss_2.detach().item(),
                0,
                actor_loss.detach().item(),
                critic_loss.detach().item(),
                self.pop_art.mu.item(),
                self.pop_art.sigma.item(),
                target_q_value_output.mean().item(),
                real_q_value.mean().detach().item(),
                r_tensor_out.mean().item())

    # do the evaluation
    def _eval_agent(self, n_epoch):
        total_success_rate = []
        for i in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset(eval=True, i_epoch=n_epoch)
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                    # noise = np.random.normal(0, 0.05, size=actions.shape)
                    # # Add noise to the action for exploration
                    # actions = (actions + noise).clip(self.env.action_space.low, self.env.action_space.high)
                if not self.args.headless_mode:
                    self.env.render()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        self.vis.line(X=[n_epoch], Y=[local_success_rate * 100], win='eval success rate', update='append')

    def eval(self):
        for i in range(2150, 2188):
            model = torch.load(f'/home/cq/code/manipulator/HER/saved_models/her_7/{i}.pt')
            self.actor_network.load_state_dict(model[-1])
            self._eval_agent(0)
