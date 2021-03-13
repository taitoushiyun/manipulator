import visdom
import numpy as np
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi
import time
# Y = np.random.rand(100)
# Y = (Y[Y > 0] + 1.5).astype(int),  # 100个标签1和2
# goal_list = []
# for goal_index in range(4):
#     goal = np.array([0.7 + 0.1 * np.cos(goal_index * np.pi / 4), 0.1 * np.sin(goal_index * np.pi / 4), 1])
#     print(goal)
#     goal_list.append(goal)
# goal_list = np.array(goal_list)

# viz = visdom.Visdom(port=6016, env='a_test')
# viz.scatter(
#     X=goal_list,
#     Y=np.ones((36, )),
#     opts={
#         'title': '3D Scatter',
#         'legend': ['Men', 'Women'],
#         'markersize': 5
#     }
# )
# time_a = time.time()
# viz.heatmap(
#     X=np.random.randint(1, 10, (200, 120)),
#     opts={
#         'columnnames': list(map(lambda x: '%.2f'% x, list(np.linspace(0.2, 1.4, num=120)))),
#         'rownames': list(map(lambda x: '%.2f'% x, list(np.linspace(0, 2, num=200)))),
#         'colormap': 'Viridis',       # 'Electric'
#     }
# )
# time_b = time.time()
# print(time_b - time_a)
# import time
# a = time.time()
# def fun():
#     return 1, 2
# data = list(map(lambda x: '%.5f'% x, list(np.linspace(0, 2, num=201, endpoint=True))))
# print(len(data))
# print(data)
# print(-1 // 0.01)
# a= np.ones((3,3))
# print(a[fun()])
# b = time.time()
# print(b-a)
# c = ['%.2f'% 0.1234]
# print(c)
# print((1.4 - 0.2) // 0.01)
# print(np.random.randint(1, 10, size=(1, 3)))

# print(DEG2RAD*RAD2DEG)
# print(np.vstack([np.ones((3, )), np.ones((3,))]))
# print(np.linalg.norm(np.array([1.2, 0, 1.2])-np.array([0.2, 0, 1]), axis=-1))
# print(np.power(np.array([1,2 ]), 0.1))
# import matplotlib.pyplot as plt
# a = np.linspace(0.01, 2, num=100)
# print(a)
# b = 1 / np.power(a, 1)
# c = 1 / np.power(a, 2)
# plt.plot(a,b, color='b')
# # plt.plot(a, c, color='r')
# plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Actor(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, actor_hidden, max_action):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             max_action (float): the maximum valid value for action
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(Actor, self).__init__()
#         self.action_size = action_size
#         self.actor_fcs = []
#         actor_in_size = state_size
#         for i, actor_next_size in enumerate(actor_hidden):
#             actor_fc = nn.Linear(actor_in_size, actor_next_size)
#             actor_in_size = actor_next_size
#             self.__setattr__("actor_fc_{}".format(i), actor_fc)
#             self.actor_fcs.append(actor_fc)
#         self.attention = nn.Sequential(nn.Linear(state_size+action_size, 64),
#                                        nn.ReLU(),
#                                        nn.Linear(64, state_size),
#                                        nn.ReLU(),
#                                        )
#         self.actor_last = nn.Linear(actor_in_size, action_size)
#         self.max_action = max_action
#
#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""
#         state = state.unsqueeze(-2).expand(*state.shape[:-1], self.action_size, state.shape[-1])
#         h = torch.cat([state, torch.eye(self.action_size).unsqueeze(0).expand((*state.shape[:-2], self.action_size, self.action_size)).to(device)], dim=-1)
#         h = self.attention(h).softmax(dim=-1)
#         h = state * h
#         for fc in self.actor_fcs:
#             h = torch.relu(fc(h))
#         h = self.actor_last(h)
#         h = h * torch.eye(self.action_size).unsqueeze(0).expand((*h.shape[:-2], self.action_size, self.action_size)).to(device)
#         h = h.sum(dim=-1)
#         return self.max_action * torch.tanh(h)



device = 'cuda'
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.action_size = env_params['action']
        self.max_action = env_params['action_max']
        self.k0 = env_params['obs'] + env_params['goal']
        self.k = 4
        self.layers = nn.ModuleList([nn.Linear(self.k0 + i * 256, 256) for i in range(self.k)])
        self.action_out = nn.Linear(256, env_params['action'])
        self.attention = nn.Sequential(nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 64),
                                       nn.ReLU(),
                                       nn.Linear(64, env_params['obs'] + env_params['goal']),
                                       nn.ReLU(),
                                       )

    def forward(self, state):
        state = state.unsqueeze(-2).expand(*state.shape[:-1], self.action_size, state.shape[-1])
        print(state.shape)
        h = torch.cat([state, torch.eye(self.action_size).unsqueeze(0).expand(
            (*state.shape[:-2], self.action_size, self.action_size)).to(device)], dim=-1)
        print(h.shape)
        h = self.attention(h).softmax(dim=-1)
        print(h.shape)
        h = state * h
        print(h.shape)
        bypass = h
        for i in range(self.k):
            x = F.relu(self.layers[i](bypass))
            bypass = torch.cat([bypass, x], -1)
        h = self.action_out(x)
        h = h * torch.eye(self.action_size).unsqueeze(0).expand((*h.shape[:-2], self.action_size, self.action_size)).to(
            device)
        h = h.sum(dim=-1)
        return self.max_action * torch.tanh(h)
env_params = {'obs': 48,
              'goal': 3,
              'action': 24,
              'action_max': 1,}

actor = actor(env_params).cuda()
input = torch.ones((2, 51)).cuda()
print(actor(input).shape)
# print(torch.ones(2,2).reshape(2,2,2))