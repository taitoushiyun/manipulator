# reference: https://raw.githubusercontent.com/henry32144/TD3-Pytorch/master/BipedalWalkerV2.ipynb
from TD3.logger import logger
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

BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.9  # discount factor
TAU = 5e-3  # for soft update of target parameters
LR_ACTOR = 1e-3  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
UPDATE_EVERY_STEP = 2  # how often to update the target and actor networks
RAND_START = 2000  # number of random exploration episodes at the start
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

    def __init__(self, state_size, action_size, max_action, fc1_units=100, fc2_units=100):
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
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.max_action = max_action

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc1_units=100, fc2_units=100):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state_action = torch.cat([state, action], dim=1)
        xs = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(xs))
        x = self.fc3(x)

        return x


class TD3Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, max_action, min_action, random_seed, noise=0.2, noise_std=0.1,
                 noise_clip=0.5):
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
        self.noise = noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, float(max_action[0])).to(device)
        self.actor_target = Actor(state_size, action_size, float(max_action[0])).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local1 = Critic(state_size, action_size).to(device)
        self.critic_target1 = Critic(state_size, action_size).to(device)
        self.critic_target1.load_state_dict(self.critic_local1.state_dict())
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=LR_CRITIC)

        self.critic_local2 = Critic(state_size, action_size).to(device)
        self.critic_target2 = Critic(state_size, action_size).to(device)
        self.critic_target2.load_state_dict(self.critic_local2.state_dict())
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=LR_CRITIC)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

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
            sigma = 1. - (1. - .05) * min(1., episode_step / 1000.)
            noise = np.random.normal(0, sigma, size=self.action_size)
            # Add noise to the action for exploration
            action = (action + noise).clip(self.min_action[0], self.max_action[0])
        self.actor_local.train()
        return action

    def learn(self, n_iteraion, gamma=GAMMA):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            n_iteraion (int): the number of iterations to train network
            gamma (float): discount factor
        """

        if len(self.memory) >= RAND_START:
            for i in range(n_iteraion):
                state, action, reward, next_state, done = self.memory.sample()

                action_ = action.cpu().numpy()

                # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models
                actions_next = self.actor_target(next_state)

                # Generate a random noise
                noise = torch.FloatTensor(action_).data.normal_(0, self.noise).to(device)
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                actions_next = (actions_next + noise).clamp(self.min_action[0].astype(float),
                                                            self.max_action[0].astype(float))

                Q1_targets_next = self.critic_target1(next_state, actions_next)
                Q2_targets_next = self.critic_target2(next_state, actions_next)

                Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
                # Compute Q targets for current states (y_i)
                Q_targets = reward + (gamma * Q_targets_next * (1 - done)).detach()
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

                if i % UPDATE_EVERY_STEP == 0:
                    # ---------------------------- update actor ---------------------------- #
                    # Compute actor loss
                    actions_pred = self.actor_local(state)
                    actor_loss = -self.critic_local1(state, actions_pred).mean()
                    # Minimize the loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ----------------------- update target networks ----------------------- #
                    self.soft_update(self.critic_local1, self.critic_target1, TAU)
                    self.soft_update(self.critic_local2, self.critic_target2, TAU)
                    self.soft_update(self.actor_local, self.actor_target, TAU)

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


def td3_torcs(env, agent, n_episodes, max_episode_length, model_dir, vis):
    os.makedirs('checkpoints/actor', exist_ok=True)
    vis.line(X=[0], Y=[0], win='result', opts=dict(Xlabel='episode', Ylabel='result', title='result'))
    vis.line(X=[0], Y=[0], win='path len', opts=dict(Xlabel='episode', Ylabel='len', title='path len'))
    vis.line(X=[0], Y=[0], win='success rate', opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='success rate'))
    vis.line(X=[0], Y=[0], win='eval result', opts=dict(Xlabel='episode', Ylabel='eval result', title='eval result'))
    vis.line(X=[0], Y=[0], win='eval path len', opts=dict(Xlabel='episode', Ylabel='len', title='eval path len'))
    vis.line(X=[0], Y=[0], win='eval success rate', opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='eval success rate'))
    result_deque = deque(maxlen=20)
    eval_result_queue = deque(maxlen=10)

    for i_episode in range(n_episodes):
        state = env.reset()
        score = 0
        episode_length = 0
        for t in count():
            if len(agent.memory) < RAND_START:
                action = env.action_space.sample()
            else:
                action = agent.act(state, episode_step=i_episode)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            episode_length += 1
            agent.learn(1)
            if done or episode_length >= max_episode_length:
                result = 0.
                if done and episode_length < max_episode_length:
                    result = 1.
                break
        result_deque.append(result)
        success_rate = np.mean(result_deque)
        logger.info(
            "Episode: %d,          Path length: %d       result: %f" % (i_episode, episode_length, result))
        # vis.line(X=[i_episode], Y=[(score - env.max_rewards) * 100], win='reward', update='append')
        # vis.line(X=[i_episode], Y=[episode_length], win='path len', update='append')
        # vis.line(X=[i_episode], Y=[(mean_score - env.max_rewards) * 100], win='mean reward', update='append')
        vis.line(X=[i_episode], Y=[result], win='result', update='append')
        vis.line(X=[i_episode], Y=[episode_length], win='path len', update='append')
        vis.line(X=[i_episode], Y=[success_rate * 100], win='success rate', update='append')
        if i_episode % 5 == 0:
            torch.save(agent.actor_local.state_dict(), os.path.join(model_dir, f'actor/{i_episode}.pth'))

            state = env.reset()
            total_reward = 0
            total_len = 0
            for t in count():
                action = agent.act(state, add_noise=False)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                total_len += 1
                state = next_state
                if done or total_len >= max_episode_length:
                    eval_result = 0
                    if done and total_len < max_episode_length:
                        eval_result = 1.
                    break
            eval_result_queue.append(eval_result)
            eval_success_rate = np.mean(eval_result_queue)
            logger.info(
                "Eval Episode: %d,          Path length: %d       result: %f" % (i_episode, total_len, eval_result))
            vis.line(X=[i_episode], Y=[eval_result], win='eval result', update='append')
            vis.line(X=[i_episode], Y=[total_len], win='eval path len', update='append')
            vis.line(X=[i_episode], Y=[eval_success_rate * 100], win='eval success rate', update='append')
            # vis.line(X=[i_episode], Y=[100 * (total_reward - env.max_rewards)], win='eval reward', update='append')
            # vis.line(X=[i_episode], Y=[total_len], win='eval path len', update='append')

