import torch
import torch.nn as nn
import numpy as np


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action, min_action,
                 max_sigma=1., min_sigma=.05, decay_period=500):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_action = max_action
        self.min_action = min_action
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        super(Policy, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(self.obs_dim, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, self.act_dim),
                                 nn.Tanh()
                                 )

    def forward(self, cur_obs_tensor):
        return self.mlp(cur_obs_tensor) * self.max_action

    def select_action(self, obs_np, episode):
        sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(1.0, episode / self._decay_period)
        action = self.forward(torch.tensor(obs_np[None], dtype=torch.float)).detach().numpy()
        noise = np.random.normal(0, sigma, size=(1, self.act_dim))
        action = (action + noise).clip(self.min_action, self.max_action)
        return action[0]

    def eval_action(self, obs_np, episode=0):
        action = self.forward(torch.tensor(obs_np[None], dtype=torch.float)).detach().numpy()
        return action[0]


class QFun(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(QFun, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mlp = nn.Sequential(nn.Linear(self.obs_dim + self.act_dim, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 1))

    def forward(self, cur_obs_tensor, actions_tensor):
        input_ = torch.cat([cur_obs_tensor, actions_tensor], 1)

        v = self.mlp(input_)
        return v







if __name__ == "__main__":
    print(np.random.randn(1, 3))

