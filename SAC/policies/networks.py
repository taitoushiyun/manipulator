import torch
import numpy as np
import abc
from torch import nn as nn
from torch.nn import functional as F
from pytorch_core.core import PyTorchModule
from pytorch_core import pytorch_util as ptu


def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(self, hidden_sizes, output_size, input_size,
                 init_w=3e-3, b_init_value=0.1,
                 hidden_activation=F.relu, output_activation=identity,
                 hidden_init=ptu.fanin_init,
                 layer_norm=False, layer_norm_kwargs=None):
        self.save_init_params(locals())
        super(Mlp, self).__init__()

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

    def forward(self, inputs, return_preactivation=False):
        h = inputs
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            # TODO: add layer normalize
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        if return_preactivation:
            return output, preactivation
        else:
            return output


class Policy(object):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy):
    def set_num_steps_total(self, t):
        pass


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self, obs_normalizer=None, *args,
            **kwargs):
        # self.save_init_params(locals())
        super(MlpPolicy, self).__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
        return super(MlpPolicy).forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class Mlp_pi(PyTorchModule):
    def __init__(self, hidden_sizes, output_size, input_size,
                 init_w=3e-3, b_init_value=0.1,
                 hidden_activation=F.relu, output_activation=torch.tanh,
                 hidden_init=ptu.fanin_init,
                 layer_norm=False, layer_norm_kwargs=None, max_sigma=1., decay_period=2000):
        self.save_init_params(locals())
        super(Mlp_pi, self).__init__()

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
        self._min_sigma = 0.05
        self._decay_period = decay_period
        self._max_sigma = max_sigma

    def forward(self, inputs, return_preactivation=False):
        h = inputs
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            # TODO: add layer normalize
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        if return_preactivation:
            return output, preactivation
        else:
            return output

    def get_action(self, obs_np, episode=0):
        sigma = (
            self._max_sigma - (self._max_sigma - self._min_sigma) *
            min(1.0, episode * 1.0 / self._decay_period)
        )
        actions = self.get_actions(obs_np[None]) + np.random.rand(1, 2) * sigma
        actions = np.clip(actions, -1., 1.)
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)

    def reset(self):
        pass


if __name__ == '__main__':

    pi = Mlp([200, 200], 3, 10)
    obs = np.ones(10)
    action = pi.forward(torch.FloatTensor(obs))
    print(torch.FloatTensor(obs))
    print(action)
    print(pi(torch.FloatTensor(obs)))
    print(pi.eval_np(obs))

    for param in pi.parameters():
        print(param.data.shape)



















