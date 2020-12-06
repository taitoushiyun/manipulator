import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



class AbstractLearner(ABC):
    def __init__(self, *, policy, batch_size):
        self.policy = policy
        self.batch_size = batch_size

    @abstractmethod
    def learn(self, **keywords):
        raise NotImplementedError


class PPOLearner(AbstractLearner):
    def __init__(self, *, policy, batch_size, minibatch_size, cliprange=0.2,
                 gamma=0.99, lr=3e-4, epochs=4, max_grad_norm=.5, entropy_coef=0.001, vf_coef=0.5):
        super(PPOLearner, self).__init__(policy=policy, batch_size=batch_size)
        self.minibatch_size = minibatch_size
        self.cliprange = cliprange
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.c = 3
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, observations, actions, values, returns, old_log_probs):
        data_size = observations.shape[0]
        for _ in range(self.epochs):
            for index in BatchSampler(SubsetRandomSampler(range(data_size)), self.minibatch_size, True):
                new_dists, policy_entropys, new_values = self.policy.compute_action(observations[index])
                new_log_probs = new_dists.log_prob(actions[index]).sum(1, keepdim=True)
                ratios = torch.exp(new_log_probs - old_log_probs[index])
                advantages = returns[index] - values[index]
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

                # print(observations.shape, old_log_probs.shape, new_log_probs.shape,
                #       returns.shape, new_values.shape, policy_entropys.shape)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1. - self.cliprange, 1. + self.cliprange) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                # policy_loss = -torch.max(torch.min(surr1, surr2), self.c * advantages.clamp(-np.infty, 0)).mean()

                # value_pre_clipped = values[index] + (new_values - values[index]).clamp(-self.cliprange, self.cliprange)
                # value_loss1 = (new_values - returns[index]).pow(2)
                # value_loss2 = (value_pre_clipped - returns[index]).pow(2)
                # value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                value_loss = (returns[index] - new_values).pow(2).mean()
                # TODO: google use an extra 0.5 and torch.max while beijing use torch.min and no extra param

                policy_ent_loss = - policy_entropys.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * policy_ent_loss
                # TODO: planning to use learning rate
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.policy.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()


def constfn(val):
    def f(_):
        return val
    return f
