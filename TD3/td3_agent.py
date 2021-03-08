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

logger = logging.getLogger('mani')

if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

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

    def __init__(self, state_size, action_size, critic_hidden):
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


class TD3Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, max_action, min_action, actor_hidden, critic_hidden, random_seed,
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
        self.actor_local = Actor(state_size, action_size, actor_hidden, float(max_action[0])).to(device)
        self.actor_target = Actor(state_size, action_size, actor_hidden, float(max_action[0])).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local1 = Critic(state_size, action_size, critic_hidden).to(device)
        self.critic_target1 = Critic(state_size, action_size, critic_hidden).to(device)
        self.critic_target1.load_state_dict(self.critic_local1.state_dict())
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=lr_critic)

        self.critic_local2 = Critic(state_size, action_size, critic_hidden).to(device)
        self.critic_target2 = Critic(state_size, action_size, critic_hidden).to(device)
        self.critic_target2.load_state_dict(self.critic_local2.state_dict())
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=lr_critic)
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory"""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, episode_step=0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if add_noise:
            # Generate a random noise
            sigma = 1. - (1. - .05) * min(1., episode_step / self.noise_drop_rate)
            noise = np.random.normal(0, sigma, size=self.action_size)
            # Add noise to the action for exploration
            action = (action + noise).clip(self.min_action[0], self.max_action[0])
        self.actor_local.train()
        return action

    def learn(self, n_iteraion):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            n_iteraion (int): the number of iterations to train network
            gamma (float): discount factor
        """

        if len(self.memory) >= self.random_start:
            for i in range(n_iteraion):
                state, action, reward, next_state, done = self.memory.sample()
                # action_ = action.cpu().numpy()

                # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models
                actions_next = self.actor_target(next_state)

                # Generate a random noise
                # noise = torch.FloatTensor(action_).data.normal_(0, self.noise).to(device)
                noise = torch.normal(torch.zeros_like(actions_next), self.noise).to(device)
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                actions_next = (actions_next + noise).clamp(self.min_action[0].astype(float),
                                                            self.max_action[0].astype(float))

                Q1_targets_next = self.critic_target1(next_state, actions_next)
                Q2_targets_next = self.critic_target2(next_state, actions_next)

                Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
                # Compute Q targets for current states (y_i)
                Q_targets = reward + (self.gamma * Q_targets_next * (1 - done)).detach()
                # Compute critic loss
                Q1_expected = self.critic_local1(state, action)
                Q2_expected = self.critic_local2(state, action)
                critic_loss1 = F.mse_loss(Q1_expected, Q_targets)
                critic_loss2 = F.mse_loss(Q2_expected, Q_targets)
                # Minimize the loss
                self.critic_optimizer1.zero_grad()
                critic_loss1.backward()
                self.critic_optimizer1.step()

                self.critic_optimizer2.zero_grad()
                critic_loss2.backward()
                self.critic_optimizer2.step()

                if i % self.update_every_step == 0:
                    # ---------------------------- update actor ---------------------------- #
                    # Compute actor loss
                    actions_pred = self.actor_local(state)
                    actor_loss = -self.critic_local1(state, actions_pred).mean()
                    # Minimize the loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ----------------------- update target networks ----------------------- #
                    self.soft_update(self.critic_local1, self.critic_target1, self.tau)
                    self.soft_update(self.critic_local2, self.critic_target2, self.tau)
                    self.soft_update(self.actor_local, self.actor_target, self.tau)

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


def td3_torcs(env, agent, n_episodes, max_episode_length, model_dir, vis, args_):
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
    result_deque = deque(maxlen=20)
    score_deque = deque(maxlen=10)
    eval_result_queue = deque(maxlen=10)
    os.makedirs(model_dir, exist_ok=True)

    if args_.load_model is not None:
        model = torch.load(args_.load_model)
        agent.actor_local.load_state_dict(model.state_dict())
        agent.actor_target.load_state_dict(model.state_dict())
        start_episode = int(args_.load_model.split('/')[-1].split('.')[0])
    else:
        start_episode = 0
    for i_episode in range(start_episode, n_episodes):
        time_a = time.time()
        state = env.reset()
        state = np.concatenate([state['observation'], state['desired_goal']])
        score = 0
        episode_length = 0
        for i in range(max_episode_length):
            if len(agent.memory) < args_.random_start:
                action = env.action_space.sample()
            else:
                action = agent.act(state, episode_step=i_episode)
            next_state, reward, done, info = env.step(action)
            next_state = np.concatenate([next_state['observation'], next_state['desired_goal']])
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            episode_length += 1
            agent.learn(1)
            if done:
                result = 0.
                if done and episode_length < max_episode_length:
                    result = 1.
                break
            # if done:
            #     result = 0.
            #     if done and episode_length < max_episode_length and not any(info['collision_state']):
            #         result = 1.
            #     break
            # if i == max_episode_length - 1:
            #     if done:
            #         result = 1.
            #     else:
            #         result = 0
        result_deque.append(result)
        score_deque.append(score)
        success_rate = np.mean(result_deque)
        mean_score = np.mean(score_deque)
        logger.info(
            "Episode: %d,          Path length: %d       result: %f       reward: %f"
            % (i_episode, episode_length, result, score))
        if args_.goal_set != 'random':
            if args_.reward_type == 'dense potential':
                vis.line(X=[i_episode], Y=[(score - env.max_rewards) * 100], win='reward', update='append')
                vis.line(X=[i_episode], Y=[(mean_score - env.max_rewards) * 100], win='mean reward', update='append')
            if args_.reward_type == 'dense distance':
                vis.line(X=[i_episode], Y=[score], win='reward', update='append')
                vis.line(X=[i_episode], Y=[mean_score], win='mean reward', update='append')
        vis.line(X=[i_episode], Y=[result], win='result', update='append')
        vis.line(X=[i_episode], Y=[episode_length], win='path len', update='append')
        vis.line(X=[i_episode], Y=[success_rate * 100], win='success rate', update='append')
        torch.save(agent.actor_local.state_dict(), os.path.join(model_dir, f'{i_episode}.pth'))
        time_b = time.time()
        print(time_b - time_a)
        if i_episode % args_.test_interval == 0:
            total_result = 0
            total_reward = 0
            for _ in range(args_.n_test_rollouts):
                state = env.reset()
                state = np.concatenate([state['observation'], state['desired_goal']])
                eval_score = 0
                total_len = 0
                for i in range(max_episode_length):
                    action = agent.act(state, add_noise=False)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.concatenate([next_state['observation'], next_state['desired_goal']])
                    eval_score += reward
                    total_len += 1
                    state = next_state
                    if done:
                        eval_result = 0.
                        if done and total_len < max_episode_length:
                            eval_result = 1.
                        break

                    # if done:
                    #     eval_result = 0.
                    #     if done and total_len < max_episode_length and not any(info['collision_state']):
                    #         eval_result = 1.
                    #     break

                    # if i == max_episode_length - 1:
                    #     if done:
                    #         eval_result = 1.
                    #     else:
                    #         eval_result = 0.
                total_result += eval_result
                total_reward += eval_score
            eval_success_rate = total_result / args_.n_test_rollouts
            eval_reward = total_reward / args_.n_test_rollouts
            logger.info(
                "Eval Epoch: %d, mean_result: %f" % (i_episode // args_.test_interval, eval_success_rate))
            vis.line(X=[i_episode // args_.test_interval], Y=[eval_success_rate * 100], win='eval success rate', update='append')
            if args_.goal_set != 'random':
                if args_.reward_type == 'dense potential':
                    vis.line(X=[i_episode // args_.test_interval], Y=[100 * (eval_reward - env.max_rewards)], win='eval reward', update='append')
                if args_.reward_type == 'dense distance':
                    vis.line(X=[i_episode // args_.test_interval], Y=[eval_reward], win='eval reward', update='append')

        # if i_episode % 5 == 0:
        #     state = env.reset()
        #     state = np.concatenate([state['observation'], state['desired_goal']])
        #     eval_score = 0
        #     total_len = 0
        #     for i in range(max_episode_length):
        #         action = agent.act(state, add_noise=False)
        #         next_state, reward, done, info = env.step(action)
        #         next_state = np.concatenate([next_state['observation'], next_state['desired_goal']])
        #         eval_score += reward
        #         total_len += 1
        #         state = next_state
        #         if done:
        #             eval_result = 0.
        #             if done and total_len < max_episode_length:
        #                 eval_result = 1.
        #             break
        #
        #         # if done:
        #         #     eval_result = 0.
        #         #     if done and total_len < max_episode_length and not any(info['collision_state']):
        #         #         eval_result = 1.
        #         #     break
        #
        #         # if i == max_episode_length - 1:
        #         #     if done:
        #         #         eval_result = 1.
        #         #     else:
        #         #         eval_result = 0.
        #     eval_result_queue.append(eval_result)
        #     eval_success_rate = np.mean(eval_result_queue)
        #     logger.info(
        #         "Eval Episode: %d,          Path length: %d       result: %f" % (i_episode, total_len, eval_result))
        #     vis.line(X=[i_episode], Y=[eval_result], win='eval result', update='append')
        #     vis.line(X=[i_episode], Y=[total_len], win='eval path len', update='append')
        #     vis.line(X=[i_episode], Y=[eval_success_rate * 100], win='eval success rate', update='append')
        #     if args_.goal_set != 'random':
        #         if args_.reward_type == 'dense potential':
        #             vis.line(X=[i_episode], Y=[100 * (eval_score - env.max_rewards)], win='eval reward', update='append')
        #         if args_.reward_type == 'dense distance':
        #             vis.line(X=[i_episode], Y=[eval_score], win='eval reward', update='append')

