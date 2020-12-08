import torch
import torch.nn as nn
from torch.distributions import Normal
MAX_LOG_STD = 2
MIN_LOG_STD = -5


class Actor_critic(nn.Module):
    def __init__(self, env, actor_obs_dims, actor_hidden_sizes, actor_action_dims,
                 critic_obs_dims, critic_hidden_sizes, critic_output_size=1,
                 actor_hidden_activation=torch.tanh, critic_hidden_activation=torch.tanh):
        super(Actor_critic, self).__init__()
        self.env = env
        self.actor_hidden_activation = actor_hidden_activation
        self.critic_hidden_activation = critic_hidden_activation
        self.actor_fcs, self.critic_fcs = [], []
        actor_in_size, critic_in_size = actor_obs_dims, critic_obs_dims

        for i, actor_next_size in enumerate(actor_hidden_sizes):
            actor_fc = nn.Linear(actor_in_size, actor_next_size)
            actor_in_size = actor_next_size
            self.__setattr__("actor_fc_{}".format(i), actor_fc)
            self.actor_fcs.append(actor_fc)
        self.actor_mean = nn.Linear(actor_in_size, actor_action_dims)
        self.actor_log_std = nn.Linear(actor_in_size, actor_action_dims)
        # self.actor_log_std = nn.Parameter(torch.zeros((1, actor_action_dims)), requires_grad=True)

        for i, critic_next_size in enumerate(critic_hidden_sizes):
            critic_fc = nn.Linear(critic_in_size, critic_next_size)
            critic_in_size = critic_next_size
            self.__setattr__("critic_fc_{}".format(i), critic_fc)
            self.critic_fcs.append(critic_fc)
        self.critic_last = nn.Linear(critic_in_size, critic_output_size)

    def forward(self, cur_obs_tensor):
        actor_h = cur_obs_tensor
        for i, actor_fc in enumerate(self.actor_fcs):
            actor_h = self.actor_hidden_activation(actor_fc(actor_h))
        mean = nn.Tanh()(self.actor_mean(actor_h))
        log_std = self.actor_log_std(actor_h).clamp(MIN_LOG_STD, MAX_LOG_STD)

        critic_h = cur_obs_tensor
        for i, critic_fc in enumerate(self.critic_fcs):
            critic_h = self.critic_hidden_activation(critic_fc(critic_h))
        critic_obs_t = self.critic_last(critic_h)

        return mean, log_std.exp(), critic_obs_t

    def select_action(self, cur_obs_tensor, max_action=1.0):
        m, std, v = self.forward(cur_obs_tensor)
        dist = Normal(m, std)
        action = dist.sample().clamp(-max_action, max_action)
        log_prob = dist.log_prob(action).sum(1)
        return action[0].numpy(), log_prob.detach(), v.detach().item()

    def compute_action(self, cur_obs_tensor):
        m, std, v = self.forward(cur_obs_tensor)
        dist = Normal(m, std)
        entropy = dist.entropy().sum(1, keepdim=True)
        return dist, entropy, v

