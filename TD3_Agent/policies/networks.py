import torch
import numpy as np
import abc
from torch import nn as nn
from torch.nn import functional as F
import sys
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/Algorithms')
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/pytorch_core')
from pytorch_core.core import PyTorchModule


class MlpPI(PyTorchModule):
    def __init__(self, hidden_sizes, act_dims, obs_dims, max_action=1.,
                 hidden_activation=F.relu, output_activation=torch.tanh,
                 min_sigma=0.05, max_sigma=1., decay_period=500):
        self.save_init_params(locals())
        super(MlpPI, self).__init__()

        self.max_action = max_action
        self._max_sigma, self._min_sigma, self._decay_period = max_sigma, min_sigma, decay_period
        self.input_size, self.output_size = obs_dims, act_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.fcs = []
        in_size = obs_dims

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size

            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, act_dims)

    def forward(self, cur_obs_tensor):
        h = cur_obs_tensor
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        pre_activation = self.last_fc(h)
        output = self.output_activation(pre_activation)

        return output * self.max_action

    def get_action(self, obs_np, episode=0):
        sigma = (
            self._max_sigma - (self._max_sigma - self._min_sigma) * min(1.0, episode / self._decay_period)
        )
        actions = self.get_actions(obs_np[None]) + np.random.randn(1, self.output_size) * sigma
        actions = np.clip(actions, -1., 1.)
        return actions[0, :], {}

    def eval_action(self, obs_np, episode=0):
        actions = self.get_actions(obs_np[None])
        actions = np.clip(actions, -1., 1.)
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)

    def reset(self):
        pass


if __name__ == '__main__':
    pi = MlpPI([64, 64], 3, 6)
    print(pi)
    obs = np.array([1., 1.1, 0.9, 2.1, 0.1, -1.])
    act = pi.get_action(obs)
    print(act)









