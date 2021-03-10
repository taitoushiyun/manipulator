import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
# class actor(nn.Module):
#     def __init__(self, env_params):
#         super(actor, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.action_out = nn.Linear(256, env_params['action'])
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         actions = self.max_action * torch.tanh(self.action_out(x))
#
#         return actions
#
#
# class critic(nn.Module):
#     def __init__(self, env_params):
#         super(critic, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.q_out = nn.Linear(256, 1)
#
#     def forward(self, x, actions):
#         x = torch.cat([x, actions / self.max_action], dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         q_value = self.q_out(x)
#
#         return q_value


# class actor(nn.Module):
#     def __init__(self, env_params):
#         super(actor, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
#         self.fc2 = nn.Linear(env_params['obs'] + env_params['goal'] + 256, 256)
#         self.fc3 = nn.Linear(env_params['obs'] + env_params['goal'] + 256, 256)
#         self.fc4 = nn.Linear(env_params['obs'] + env_params['goal'] + 256, 256)
#         self.action_out = nn.Linear(256, env_params['action'])
#
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#
#         x = torch.cat([x, state], -1)
#         x = F.relu(self.fc2(x))
#
#         x = torch.cat([x, state], -1)
#         x = F.relu(self.fc3(x))
#
#         x = torch.cat([x, state], -1)
#         x = F.relu(self.fc4(x))
#
#         actions = self.max_action * torch.tanh(self.action_out(x))
#
#         return actions
#
#
# class critic(nn.Module):
#     def __init__(self, env_params):
#         super(critic, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
#         self.fc2 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] + 256, 256)
#         self.fc3 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] + 256, 256)
#         self.fc4 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] + 256, 256)
#         self.q_out = nn.Linear(256, 1)
#
#     def forward(self, state, actions):
#         sa = torch.cat([state, actions / self.max_action], dim=1)
#         x = F.relu(self.fc1(sa))
#
#         x = torch.cat([x, sa], -1)
#         x = F.relu(self.fc2(x))
#
#         x = torch.cat([x, sa], -1)
#         x = F.relu(self.fc3(x))
#
#         x = torch.cat([x, sa], -1)
#         x = F.relu(self.fc4(x))
#         q_value = self.q_out(x)
#
#         return q_value

class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
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


class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
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
        q_value = self.q_out(x)
        return q_value