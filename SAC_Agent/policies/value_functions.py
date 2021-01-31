import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from SAC_Agent.pytorch_core import pytorch_util as ptu
from SAC_Agent.pytorch_core.core import PyTorchModule
from SAC_Agent.policies.networks import identity


class VFunction(PyTorchModule):
    def __init__(self, hidden_sizes, output_size, input_size,
                 init_w=3e-3, b_init_value=0.1,
                 hidden_activation=F.relu, output_activation=identity,
                 hidden_init=ptu.fanin_init,
                 layer_norm=False, layer_norm_kwargs=None):

        self.save_init_params(locals())
        super(VFunction, self).__init__()

        assert output_size == 1

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size, self.output_size = input_size, output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.layer_norm = layer_norm
        self.fcs, self.layer_norms = [], []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size

            # initialize parameters of W and b
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)

            self.fcs.append(fc)
            # TODO: add layer normalize

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, observation):
        h = observation
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            # TODO: add layer normalize
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        return output


class QFunction(PyTorchModule):
    def __init__(self, hidden_sizes, output_size, input_size,
                 init_w=3e-3, b_init_value=0.1,
                 hidden_activation=F.relu, output_activation=identity,
                 hidden_init=ptu.fanin_init,
                 layer_norm=False, layer_norm_kwargs=None):
        self.save_init_params(locals())
        super(QFunction, self).__init__()

        assert output_size == 1

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size, self.output_size = input_size, output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.layer_norm = layer_norm
        self.fcs, self.layer_norms = [], []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size

            # initialize parameters of W and b
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)

            self.fcs.append(fc)
            # TODO: add layer normalize

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, observation, action):
        h = torch.cat([observation, action], -1)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            # TODO: add layer normalize
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        return output




















