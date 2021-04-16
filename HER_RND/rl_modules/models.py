import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""


class Actor(nn.Module):
    def __init__(self, env_params, args):
        super(Actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions


class Critic(nn.Module):
    def __init__(self, env_params, args):
        super(Critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


class ActorDenseSimple(nn.Module):
    def __init__(self, env_params, args):
        super(ActorDenseSimple, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(env_params['obs'] + env_params['goal'] + 256, 256)
        self.fc3 = nn.Linear(env_params['obs'] + env_params['goal'] + 256, 256)
        self.fc4 = nn.Linear(env_params['obs'] + env_params['goal'] + 256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, state):
        x = F.relu(self.fc1(state))

        x = torch.cat([x, state], -1)
        x = F.relu(self.fc2(x))

        x = torch.cat([x, state], -1)
        x = F.relu(self.fc3(x))

        x = torch.cat([x, state], -1)
        x = F.relu(self.fc4(x))

        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class CriticDenseSimple(nn.Module):
    def __init__(self, env_params, args):
        super(CriticDenseSimple, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] + 256, 256)
        self.fc3 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] + 256, 256)
        self.fc4 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] + 256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, state, actions):
        sa = torch.cat([state, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(sa))

        x = torch.cat([x, sa], -1)
        x = F.relu(self.fc2(x))

        x = torch.cat([x, sa], -1)
        x = F.relu(self.fc3(x))

        x = torch.cat([x, sa], -1)
        x = F.relu(self.fc4(x))
        q_value = self.q_out(x)

        return q_value


class ActorDense(nn.Module):
    def __init__(self, env_params, args):
        super(ActorDense, self).__init__()
        self.max_action = env_params['action_max']
        self.k0 = env_params['obs'] + env_params['goal']
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, state):
        bypass = state
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class ActorDenseASF(nn.Module):
    def __init__(self, env_params, args):
        super(ActorDenseASF, self).__init__()
        self.args = args
        if self.args.cuda:
            self.local_device = 'cuda'
        else:
            self.local_device = 'cpu'
        self.action_size = env_params['action']
        self.max_action = env_params['action_max']
        self.k0 = env_params['obs'] + env_params['goal']
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.action_out = nn.Linear(256, 1)
        self.attention = nn.Sequential(nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 16),
                                       nn.ReLU(),
                                       # nn.Linear(64, 64),
                                       # nn.ReLU(),
                                       nn.Linear(16, env_params['obs'] + env_params['goal']),
                                       nn.ReLU(),
                                       )

    def forward(self, state):
        state = state.unsqueeze(-2).expand(*state.shape[:-1], self.action_size, state.shape[-1])
        h = torch.cat([state, torch.eye(self.action_size).unsqueeze(0).expand(
            (*state.shape[:-2], self.action_size, self.action_size)).to(self.local_device)], dim=-1)
        h = self.attention(h).softmax(dim=-1)
        h = state * h
        bypass = h
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        h = self.action_out(x)
        h = h * torch.eye(self.action_size).unsqueeze(0).expand((*h.shape[:-2], self.action_size, self.action_size)).to(
            self.local_device)
        h = h.sum(dim=-1)
        # h = h.squeeze(-1)
        return self.max_action * torch.tanh(h)


class CriticDense(nn.Module):
    def __init__(self, env_params, args):
        super(CriticDense, self).__init__()
        self.max_action = env_params['action_max']
        self.k0 = env_params['obs'] + env_params['goal'] + env_params['action']
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.q_out = nn.Linear(256, 1)

        self.mu = self.mu_new = torch.tensor(0)
        self.sigma = self.sigma_new = torch.tensor(1)
        self.nu = torch.tensor(0)
        self.beta = args.beta
        self.cnt = 0

    def forward(self, state, actions):
        bypass = torch.cat([state, actions / self.max_action], dim=1)
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        q_value = self.q_out(x)
        return q_value


class DNet(nn.Module):
    def __init__(self, env_params, args):
        super(DNet, self).__init__()
        self.max_action = env_params['action_max']
        self.k0 = env_params['obs'] + env_params['action']
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.q_out = nn.Linear(256, env_params['obs'])

    def forward(self, state, actions):
        bypass = torch.cat([state, actions / self.max_action], dim=1)
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        obs_predict = self.q_out(x)
        return obs_predict

    def compute_loss(self, state, action, next_state):
        loss = (self.forward(state, action) - next_state.detach()).pow(2)
        return loss


class RND(nn.Module):
    def __init__(self, env_params, args):
        super(RND, self).__init__()
        self.max_action = env_params['action_max']
        self.k0 = env_params['obs']
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.q_out = nn.Linear(256, 256)

    def forward(self, state):
        bypass = state
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        obs_predict = self.q_out(x)
        return obs_predict


class Dynamic(nn.Module):
    def __init__(self, env_params, args):
        super(Dynamic, self).__init__()
        self.target_net = RND(env_params, args)
        self.predict_net = RND(env_params, args)

    def compute_loss(self, state, action, next_action):
        loss = (self.target_net(next_action).detach() - self.predict_net(next_action)).pow(2)
        return loss


class CriticLowerLayer(nn.Module):
    def __init__(self, env_params):
        super(CriticLowerLayer, self).__init__()
        self.max_action = env_params['action_max']
        self.k0 = env_params['obs'] + env_params['goal'] + env_params['action']
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.q_out = nn.Linear(256, 1)

    def forward(self, state, actions):
        bypass = torch.cat([state, actions / self.max_action], dim=1)
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        h_theta = self.q_out(x)
        return h_theta


class CriticUpperLayer(nn.Module):
    def __init__(self, env_params):
        super(CriticUpperLayer, self).__init__()
        self.fc = nn.Linear(1, 1)
        nn.init.ones_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input):
        normalized_pred = self.fc(input)
        return normalized_pred


class PopArtCritic(nn.Module):
    def __init__(self, env_params, args):
        super(PopArtCritic, self).__init__()
        self.lower_layer = CriticLowerLayer(env_params)
        self.upper_layer = CriticUpperLayer(env_params)

    def forward(self, state, actions):
        normalized_pred = self.upper_layer(self.lower_layer(state, actions))
        return normalized_pred


