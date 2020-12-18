import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
import sys
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/Algorithms')
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/pytorch_core')
from pytorch_core.core import PyTorchModule


class QFunction(PyTorchModule):
    def __init__(self, hidden_sizes, obs_dims, act_dims, output_size=1, hidden_activation=F.relu):
        self.save_init_params(locals())
        super(QFunction, self).__init__()
        assert output_size == 1
        self.input_size, self.output_size = obs_dims + act_dims, output_size
        self.hidden_activation = hidden_activation
        self.fcs = []
        in_size = self.input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, cur_obs_tensor, cur_act_tensor):
        h = torch.cat([cur_obs_tensor, cur_act_tensor], -1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        pre_activation = self.last_fc(h)
        return pre_activation


class VFunction(PyTorchModule):
    def __init__(self, hidden_sizes, obs_dims, output_size=1, hidden_activation=F.relu,):
        self.save_init_params(locals())
        super(VFunction, self).__init__()
        assert output_size == 1
        self.input_size, self.output_size = obs_dims, output_size
        self.hidden_activation = hidden_activation
        self.fcs = []
        in_size = self.input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, cur_obs_tensor):
        h = cur_obs_tensor
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        pre_activation = self.last_fc(h)
        return pre_activation


if __name__ == '__main__':
    V = VFunction([64, 64], 5)
    print(V)
    obs = torch.FloatTensor([1., 1.1, 1.2, 1.3, 1.4])
    v = V(obs[None])
    print(v)











