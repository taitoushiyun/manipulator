import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from algorithms.rl_algorithm import RLAlgorithm
import pytorch_core.pytorch_util as ptu


class TD3(RLAlgorithm):
    def __init__(self, base_kwargs, env, policy, pool,
                 qf1, qf2, target_policy_noise=0.2,
                 target_policy_noise_clip=0.5, tau=0.005,
                 policy_and_target_update_period=2, discount=0.99):
        super(TD3, self).__init__(**base_kwargs)

        self.env, self.policy, self.pool = env, policy, pool
        self.qf1, self.qf2 = qf1, qf2
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.tau = tau
        self._n_train_steps_total = 0
        self.discount = discount
        self.policy_and_target_update_period = policy_and_target_update_period

        self.qf_criterion = nn.MSELoss()
        self.qf1_optimizer = optim.Adam(self.qf1.parameters())
        self.qf2_optimizer = optim.Adam(self.qf2.parameters())
        self.policy_optimizer = optim.Adam(self.policy.parameters())
        self.qf1_target = self.qf1.copy()
        self.qf2_target = self.qf2.copy()
        self.target_policy = policy.copy()

    def train(self, ):
        self._train(self.env, self.policy, self.pool, save_param_name='td3_')

    def _do_training(self, num_iter, batch_data):
        rewards = batch_data['rewards']
        terminals = batch_data['terminals']
        obs = batch_data['observations']
        actions = batch_data['actions']

        next_obs = batch_data['next_observations']

        next_actions = self.target_policy(next_obs)

        noise = torch.normal(torch.zeros_like(next_actions),
                             self.target_policy_noise)
        noise = torch.clamp(noise, -self.target_policy_noise_clip,
                            self.target_policy_noise_clip)

        noisy_next_actions = torch.clamp(next_actions + noise, -1., 1.)
        # noisy_next_actions = next_actions + noise
        qf1 = self.qf1_target(next_obs, noisy_next_actions)
        qf2 = self.qf2_target(next_obs, noisy_next_actions)
        v_preds = torch.min(qf1, qf2)
        q_targets = rewards + (1. - terminals) * self.discount * v_preds

        qf1_loss = self.qf_criterion(self.qf1(obs, actions), q_targets.detach())
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        qf2_loss = self.qf_criterion(self.qf2(obs, actions), q_targets.detach())
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            new_actions = self.policy(obs)
            q_loss = -self.qf1(obs, new_actions).mean()
            self.policy_optimizer.zero_grad()
            q_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
            ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)

        self._n_train_steps_total += 1

    def _do_evaluate(self):
        pass



























