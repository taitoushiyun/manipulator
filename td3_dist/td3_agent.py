# reference: https://raw.githubusercontent.com/henry32144/TD3-Pytorch/master/BipedalWalkerV2.ipynb
import gym
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import namedtuple, deque
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import count
import logging
import time
from normalizer import Normalizer_torch2
from pop_art import PopArt
from models import DNet, Dynamic
from mujoco.env import ManipulatorEnv

logger = logging.getLogger('mani')

if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        for i in range(done.shape[0]):
            self.memory.append(self.experience(state[i], action[i], reward[i], next_state[i], done[i]))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, actor_hidden, max_action):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): the maximum valid value for action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.actor_fcs = []
        actor_in_size = state_size
        for i, actor_next_size in enumerate(actor_hidden):
            actor_fc = nn.Linear(actor_in_size, actor_next_size)
            actor_in_size = actor_next_size
            self.__setattr__("actor_fc_{}".format(i), actor_fc)
            self.actor_fcs.append(actor_fc)
        self.actor_last = nn.Linear(actor_in_size, action_size)
        self.max_action = max_action

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        h = state
        for fc in self.actor_fcs:
            h = torch.relu(fc(h))
        return self.max_action * torch.tanh(self.actor_last(h))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, critic_hidden, max_action):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.critic_fcs = []
        critic_in_size = state_size + action_size
        for i, critic_next_size in enumerate(critic_hidden):
            critic_fc = nn.Linear(critic_in_size, critic_next_size)
            critic_in_size = critic_next_size
            self.__setattr__("critic_fc_{}".format(i), critic_fc)
            self.critic_fcs.append(critic_fc)
        self.critic_last = nn.Linear(critic_in_size, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state_action = torch.cat([state, action], dim=1)
        h = state_action
        for fc in self.critic_fcs:
            h = torch.relu(fc(h))
        return self.critic_last(h)


class ActorDense(nn.Module):
    def __init__(self, state_size, action_size, actor_hidden, max_action):
        super(ActorDense, self).__init__()
        self.max_action = max_action
        self.k0 = state_size
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.action_out = nn.Linear(256, action_size)

    def forward(self, state):
        bypass = state
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class CriticDense(nn.Module):
    def __init__(self, state_size, action_size, critic_hidden, max_action):
        super(CriticDense, self).__init__()
        self.max_action = max_action
        self.k0 = state_size + action_size
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.q_out = nn.Linear(256, 1)

    def forward(self, state, actions):
        bypass = torch.cat([state, actions / self.max_action], dim=1)
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        q_value = self.q_out(x)
        return q_value

def normalize(y, pop_art):
    if pop_art is None:
        return y
    return (y - pop_art.mu) / pop_art.sigma


def denormalize(y, pop_art):
    if pop_art is None:
        return y
    return pop_art.sigma * y + pop_art.mu

class TD3Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, args, env, state_size, action_size, max_action, min_action, actor_hidden, critic_hidden, random_seed,
                 gamma=0.99, tau=5e-3, lr_actor=1e-3, lr_critic=1e-3, update_every_step=2, random_start=2000,
                 noise=0.2, noise_std=0.1, noise_clip=0.5, noise_drop_rate=500.,
                 buffer_size=int(1e7), batch_size=64):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            max_action (ndarray): the maximum valid value for each action vector
            min_action (ndarray): the minimum valid value for each action vector
            random_seed (int): random seed
            noise (float): the range to generate random noise while learning
            noise_std (float): the range to generate random noise while performing action
            noise_clip (float): to clip random noise into this range
        """
        self.args = args
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.min_action = min_action
        self.gamma = gamma
        self.tau = tau
        self.update_every_step = update_every_step
        self.random_start = random_start
        self.noise = noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.noise_drop_rate = noise_drop_rate
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        if args.actor_type == 'mlp':
            actor = Actor
        elif args.actor_type == 'dense':
            actor = ActorDense
        else:
            raise ValueError
        if args.critic_type == 'mlp':
            critic = Critic
        elif args.critic_type == 'dense':
            critic = CriticDense
        else:
            raise ValueError
        self.actor_local = actor(state_size+3, action_size, actor_hidden, float(max_action)).to(device)
        self.actor_eval = actor(state_size + 3, action_size, actor_hidden, float(max_action)).to(device)
        self.actor_eval.load_state_dict(self.actor_local.state_dict())
        self.actor_target = actor(state_size+3, action_size, actor_hidden, float(max_action)).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.actor_eval_optimizer = optim.Adam(self.actor_eval.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local1 = critic(state_size+3, action_size, critic_hidden, float(max_action)).to(device)
        self.critic_target1 = critic(state_size+3, action_size, critic_hidden, float(max_action)).to(device)
        self.critic_target1.load_state_dict(self.critic_local1.state_dict())
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=lr_critic)

        self.critic_local2 = critic(state_size+3, action_size, critic_hidden, float(max_action)).to(device)
        self.critic_target2 = critic(state_size+3, action_size, critic_hidden, float(max_action)).to(device)
        self.critic_target2.load_state_dict(self.critic_local2.state_dict())
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=lr_critic)

        if self.args.action_q:
            self.critic_action = critic(state_size+3, action_size, critic_hidden, float(max_action)).to(device)
            self.critic_action_target = critic(state_size+3, action_size, critic_hidden, float(max_action)).to(device)
            self.critic_action_target.load_state_dict(self.critic_action.state_dict())
            self.critic_action_optimizer = optim.Adam(self.critic_action.parameters(), lr=lr_critic)
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
        if self.args.use_rms_reward:
            self.r_explore_norm = Normalizer_torch2(device=device, size=1, default_clip_range=self.args.clip_range)
        else:
            self.r_explore_norm = None

        env_params = {'obs': self.state_size,
                      'goal': 3,
                      'action': self.action_size,
                      'action_max': self.max_action,
                     }
        if self.args.curiosity_type == 'forward':
            self.predict_network = DNet(env_params, args).to(device)
        elif self.args.curiosity_type == 'rnd':
            self.predict_network = Dynamic(env_params, args).to(device)
        else:
            raise ValueError('curiosity type must be forward or rnd')
        self.critic_explore_network = critic(state_size+3, action_size, critic_hidden, float(max_action)).to(device)
        self.critic_explore_target_network = critic(state_size+3, action_size, critic_hidden, float(max_action)).to(device)
        self.critic_explore_target_network.load_state_dict(self.critic_explore_network.state_dict())
        self.critic_explore_optim = torch.optim.Adam(self.critic_explore_network.parameters(),
                                                     lr=self.args.lr_critic_explore)
        self.predict_optim = torch.optim.Adam(self.predict_network.parameters(), lr=self.args.lr_predict)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory"""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, episode_step=0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            if add_noise:
                action = self.actor_local(state).cpu().data.numpy()
            else:
                action = self.actor_eval(state).cpu().data.numpy()
        if add_noise:
            # Generate a random noise
            sigma = 1. - (1. - .05) * min(1., episode_step / self.noise_drop_rate)
            noise = np.random.normal(0, sigma, size=(self.args.nenvs, self.action_size))
            # Add noise to the action for exploration
            action = (action + noise).clip(self.min_action, self.max_action)
        self.actor_local.train()
        return action

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            n_iteraion (int): the number of iterations to train network
            gamma (float): discount factor
        """

        if len(self.memory) >= self.random_start:
            state, action, reward, next_state, done = self.memory.sample()
            # action_ = action.cpu().numpy()
            predict_loss_ = self.predict_network.compute_loss(state[..., :-3], action,
                                                              next_state[..., :-3])
            r_explore_tensor = predict_loss_.mean(dim=-1, keepdim=True).detach()
            predict_loss = predict_loss_.mean()
            self.predict_optim.zero_grad()
            predict_loss.backward()
            self.predict_optim.step()

            if self.args.use_rms_reward:
                self.r_explore_norm.update(r_explore_tensor)



            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_state)

            q_explore_next_value = self.critic_explore_target_network(next_state, actions_next).detach()
            r_explore_tensor = denormalize(r_explore_tensor, self.r_explore_norm)
            target_q_explore_value = r_explore_tensor + self.args.gamma * q_explore_next_value
            target_q_explore_value = target_q_explore_value.detach()
            target_q_explore_value_output = target_q_explore_value.mean()
            r_explore_tensor_output = r_explore_tensor.mean()
            # q explore loss
            predicted_q_explore_value = self.critic_explore_network(state, action)
            # predicted_q_explore_value_output = predicted_q_explore_value.mean()
            critic_explore_loss = (target_q_explore_value.detach() - predicted_q_explore_value).pow(2).mean()

            # Generate a random noise
            # noise = torch.FloatTensor(action_).data.normal_(0, self.noise).to(device)
            noise = torch.normal(torch.zeros_like(actions_next), self.noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            actions_next = (actions_next + noise).clamp(self.min_action.astype(float),
                                                        self.max_action.astype(float))

            Q1_targets_next = self.critic_target1(next_state, actions_next)
            Q2_targets_next = self.critic_target2(next_state, actions_next)
            if self.args.action_q:
                # next_joint_state = state[::, self.env.j_ang_idx] + DEG2RAD * 0.2 * self.args.max_angles_vel * action
                next_joint_state = next_state[::, self.env.j_ang_idx]
                action_q_target = torch.pow(next_joint_state / 1.57, 2).mean().detach()

            Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
            # Compute Q targets for current states (y_i)
            Q_targets = reward + (self.gamma * Q_targets_next * (1 - done)).detach()
            # Compute critic loss
            Q1_expected = self.critic_local1(state, action)
            Q2_expected = self.critic_local2(state, action)
            critic_loss1 = F.mse_loss(Q1_expected, Q_targets)
            critic_loss2 = F.mse_loss(Q2_expected, Q_targets)
            if self.args.action_q:
                Q_action_expected = self.critic_action(state, action)
                critic_action_loss = F.mse_loss(Q_action_expected, action_q_target)
            # Minimize the loss
            self.critic_optimizer1.zero_grad()
            critic_loss1.backward()
            self.critic_optimizer1.step()

            self.critic_optimizer2.zero_grad()
            critic_loss2.backward()
            self.critic_optimizer2.step()

            self.critic_explore_optim.zero_grad()
            critic_explore_loss.backward()
            self.critic_explore_optim.step()


            if self.args.action_q:
                self.critic_action_optimizer.zero_grad()
                critic_action_loss.backward()
                self.critic_action_optimizer.step()

            if i % self.update_every_step == 0:
                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                actions_pred = self.actor_local(state)
                actor_loss_1 = -self.critic_local1(state, actions_pred).mean()
                actor_loss_2 = -self.critic_explore_network(state, actions_pred).mean() * self.args.q_explore_weight
                actor_loss = actor_loss_1 + actor_loss_2

                actions_eval_pred = self.actor_eval(state)
                actor_eval_loss = -self.critic_local1(state, actions_eval_pred).mean()

                # Minimize the loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.actor_eval_optimizer.zero_grad()
                actor_eval_loss.backward()
                self.actor_eval_optimizer.step()

                # ----------------------- update target networks ----------------------- #
                self.soft_update(self.critic_local1, self.critic_target1, self.tau)
                self.soft_update(self.critic_local2, self.critic_target2, self.tau)
                if self.args.action_q:
                    self.soft_update(self.critic_action, self.critic_action_target, self.tau)
                self.soft_update(self.actor_local, self.actor_target, self.tau)
            return (critic_loss1.detach().item(),
                    actor_loss.detach().item(),
                    actor_eval_loss.detach().item(),
                    critic_explore_loss.detach().item(),
                    target_q_explore_value_output.detach().item(),
                    r_explore_tensor_output.detach().item())
        else:
            time.sleep(0.5)
            return 0., 0., 0., 0., 0., 0.

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def batch_process(obs):
    if len(obs.shape) > 3:
        obs = obs.swapaxes(1, 2)
        obs = obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:])
    return obs

class TD3Worker():
    """Interacts with and learns from the environment."""

    def __init__(self, args, env, state_size, action_size, max_action, min_action, actor_hidden, critic_hidden, random_seed,
                 gamma=0.99, tau=5e-3, lr_actor=1e-3, lr_critic=1e-3, update_every_step=2, random_start=2000,
                 noise=0.2, noise_std=0.1, noise_clip=0.5, noise_drop_rate=500.,
                 buffer_size=int(1e7), batch_size=64):
        self.args = args
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.min_action = min_action
        self.gamma = gamma
        self.tau = tau
        self.update_every_step = update_every_step
        self.random_start = random_start
        self.noise = noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.noise_drop_rate = noise_drop_rate
        self.seed = random.seed(random_seed)

        if args.actor_type == 'mlp':
            actor = Actor
        elif args.actor_type == 'dense':
            actor = ActorDense
        else:
            raise ValueError
        self.actor = actor(state_size+3, action_size, actor_hidden, float(max_action)).to(device)

    def act(self, state, episode_step=0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        if add_noise:
            # Generate a random noise
            sigma = 1. - (1. - .05) * min(1., episode_step / self.noise_drop_rate)
            noise = np.random.normal(0, sigma, size=(self.args.nenvs, self.action_size))
            # Add noise to the action for exploration
            action = (action + noise).clip(self.min_action, self.max_action)
        self.actor.train()
        return action

def queue_worker(lock, queue, buffer):
    while True:
        if not queue.empty():
            with lock:
                buffer.add(*queue.get())
        else:
            time.sleep(0.1)

def run_worker(index, args_, env_config, env_params, policy, eval_policy, lock, eval_lock, queue, eval_queue):
    env = ManipulatorEnv(env_config)
    env.action_space.seed(index)
    np.random.seed(index)
    random.seed(index)
    torch.manual_seed(index)
    # if index == 0:
    #     worker_type = 'run'
    # else:
    #     worker_type = 'eval'
    worker_type = 'run'
    agent = TD3Worker(args=args_,
                      env=env,
                      state_size=env_params['obs'],
                      action_size=env_params['action'],
                      max_action=env_params['action_max'],
                      min_action=-env_params['action_max'],
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
    if worker_type == 'run':
        while True:
            agent.actor.load_state_dict(policy.state_dict())
            obs_buffer, action_buffer, reward_buffer, done_buffer, next_obs_buffer = [], [], [], [], []
            state_raw = env.reset()
            state = np.concatenate([state_raw['observation'], state_raw['desired_goal']], -1)
            for i in range(50):
                print(i)
                if len(agent.memory) < args_.random_start:
                    action = env.action_space.sample()
                else:
                    action = agent.actor(state, add_noise=True)
                next_state_raw, reward, done, info = env.step(action)
                next_state = np.concatenate([next_state_raw['observation'], next_state_raw['desired_goal']], -1)
                obs_buffer.append(state)
                action_buffer.append(action)
                reward_buffer.append(reward)
                done_buffer.append(done)
                next_obs_buffer.append(next_state)
            with lock:
                queue.put([obs_buffer, action_buffer, reward_buffer, next_obs_buffer, done_buffer])
    elif worker_type == 'eval':
        while True:
            agent.actor.load_state_dict(eval_policy.state_dict())
            obs_buffer, action_buffer, reward_buffer, done_buffer, next_obs_buffer = [], [], [], [], []
            obs = env.reset()
            for i in range(50):
                action = agent.actor(obs, add_noise=True)
                next_obs, reward, done, info = env.step(action)
            with eval_lock:
                eval_queue.put([obs_buffer, action_buffer, reward_buffer, done_buffer, next_obs_buffer])


def td3_torcs(env, agent, n_episodes, max_episode_length, model_dir, vis, args_):
    if args_.load_model is None:
        vis.line(X=[0], Y=[0], win='result', opts=dict(Xlabel='episode', Ylabel='result', title='result'))
        vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
        vis.line(X=[0], Y=[0], win='success rate', opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='success rate'))
        vis.line(X=[0], Y=[0], win='eval result', opts=dict(Xlabel='episode', Ylabel='eval result', title='eval result'))
        vis.line(X=[0], Y=[0], win='eval path len', opts=dict(Xlabel='episode', Ylabel='len', title='eval path len'))
        vis.line(X=[0], Y=[0], win='eval success rate', opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='eval success rate'))
        if args_.goal_set != 'random':
            vis.line(X=[0], Y=[0], win='reward', opts=dict(Xlabel='episode', Ylabel='reward', title='reward'))
            vis.line(X=[0], Y=[0], win='mean reward', opts=dict(Xlabel='episode', Ylabel='reward', title='mean reward'))
            vis.line(X=[0], Y=[0], win='eval reward', opts=dict(Xlabel='episode', Ylabel='reward', title='eval reward'))

        vis.line(X=[0], Y=[0], win='critic_loss', opts=dict(title='critic_loss'))
        vis.line(X=[0], Y=[0], win='actor_loss', opts=dict(title='actor_loss'))
        vis.line(X=[0], Y=[0], win='actor_eval_loss', opts=dict(title='actor_eval_loss'))
        vis.line(X=[0], Y=[0], win='critic_explore_loss', opts=dict(title='critic_explore_loss'))
        vis.line(X=[0], Y=[0], win='target_q_explore', opts=dict(title='target_q_explore'))
        vis.line(X=[0], Y=[0], win='r_explore', opts=dict(title='r_explore'))
    result_deque = deque(maxlen=20)
    score_deque = deque(maxlen=10)
    eval_result_queue = deque(maxlen=10)
    os.makedirs(model_dir, exist_ok=True)


    for i_episode in range(n_episodes):
        critic_loss, actor_loss, actor_eval_loss, critic_explore_loss, target_q_explore, r_explore = agent.learn()
        vis.line(X=[i_episode], Y=[critic_loss], win='critic_loss', update='append')
        vis.line(X=[i_episode], Y=[actor_loss], win='actor_loss', update='append')
        vis.line(X=[i_episode], Y=[actor_eval_loss], win='actor_eval_loss', update='append')
        vis.line(X=[i_episode], Y=[critic_explore_loss], win='critic_explore_loss', update='append')
        vis.line(X=[i_episode], Y=[target_q_explore], win='target_q_explore', update='append')
        vis.line(X=[i_episode], Y=[r_explore], win='r_explore', update='append')
        print(i_episode)

        # if args_.goal_set != 'random':
        #     if args_.reward_type == 'dense potential':
        #         vis.line(X=[i_episode], Y=[(score - env.max_rewards) * 100], win='reward', update='append')
        #         vis.line(X=[i_episode], Y=[(mean_score - env.max_rewards) * 100], win='mean reward', update='append')
        #     if args_.reward_type == 'dense distance':
        #         vis.line(X=[i_episode], Y=[score], win='reward', update='append')
        #         vis.line(X=[i_episode], Y=[mean_score], win='mean reward', update='append')
        # vis.line(X=[i_episode], Y=[result], win='result', update='append')
        # vis.line(X=[i_episode], Y=[episode_length], win='path len', update='append')
        # vis.line(X=[i_episode], Y=[success_rate * 100], win='success rate', update='append')
        #
        # if i_episode % 10 == 0:
        #     if i_episode > 0.9 * n_episodes:
        #         torch.save(agent.actor_eval.state_dict(), os.path.join(model_dir, f'{i_episode}.pth'))
        #     else:
        #         torch.save(agent.actor_eval.state_dict(), os.path.join(model_dir, 'model.pth'))
        # if i_episode % 5 == 0:
        #     eval_result_queue.append(eval_result)
        #     eval_success_rate = np.mean(eval_result_queue)
        #
        #     vis.line(X=[i_episode], Y=[eval_result], win='eval result', update='append')
        #     vis.line(X=[i_episode], Y=[total_len], win='eval path len', update='append')
        #     vis.line(X=[i_episode], Y=[eval_success_rate * 100], win='eval success rate', update='append')
        #     if args_.goal_set != 'random':
        #         if args_.reward_type == 'dense potential':
        #             vis.line(X=[i_episode], Y=[100 * (eval_score - env.max_rewards)], win='eval reward', update='append')
        #         if args_.reward_type == 'dense distance':
        #             vis.line(X=[i_episode], Y=[eval_score], win='eval reward', update='append')

