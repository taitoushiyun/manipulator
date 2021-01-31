import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from algorithms.rl_algorithm import RLAlgorithm
import pytorch_core.pytorch_util as ptu


class SAC(RLAlgorithm):
    def __init__(self, base_kwargs,
                 env, policy, pool,
                 qf, vf,
                 discount=0.99, soft_target_tau=1e-2,
                 optimizer_class=optim.Adam, pi_lr=1e-3, qf_lr=1e-3, vf_lr=1e-3,
                 policy_mean_reg_weight=1e-3, policy_std_reg_weight=1e-3,
                 policy_pre_activation_weight=0.,
                 train_policy_with_reparameterization=True,
                 use_automatic_entropy_tuning=True,
                 target_entropy=None):

        super(SAC, self).__init__(**base_kwargs)

        self.env, self.policy, self.pool = env, policy, pool
        self.qf, self.vf = qf, vf
        self.target_vf = vf.copy()

        self.pi_optimizer = optimizer_class(self.policy.parameters(), lr=pi_lr)
        self.qf_criterion = nn.MSELoss()
        self.qf_optimizer = optimizer_class(self.qf.parameters(), lr=qf_lr)
        self.vf_criterion = nn.MSELoss()
        self.vf_optimizer = optimizer_class(self.vf.parameters(), lr=vf_lr)

        self.discount = discount
        self.soft_target_tau = soft_target_tau

        self.train_policy_with_reparameterization = train_policy_with_reparameterization
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(1).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=pi_lr)

    def train(self, load=None):
        self._train(self.env, self.policy, self.pool, load=load)
        # print('Show the performance...')
        # self.env.show_gif()

    def _do_training(self, num_iter, batch_data):
        obs, actions, next_obs = batch_data['observations'], batch_data['actions'], batch_data['next_observations']
        rewards, terminals = batch_data['rewards'], batch_data['terminals']

        q_pred = self.qf(obs, actions)
        v_pred = self.vf(obs)
        policy_outputs = self.policy(obs, reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        """                     Q function loss                             """
        target_v_values = self.target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """                     V function loss                             """
        q_new_actions = self.qf(obs, new_actions)
        v_target = q_new_actions - alpha * log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """                    pi function loss                             """
        if self.train_policy_with_reparameterization:
            pi_loss = (alpha * log_pi - q_new_actions).mean()
        else:
            log_pi_target = q_new_actions - v_pred
            pi_loss = (log_pi * (alpha * log_pi - log_pi_target).detach()).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * ((pre_tanh_value ** 2).sum(dim=1).mean())
        pi_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss

        policy_loss = pi_loss + pi_reg_loss
        """                    update Q, V and pi                            """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

        self._update_target_network()

        # TODO: Record some statistics of training process for evaluating.

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _do_evaluate(self):
        pass
































