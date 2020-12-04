import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions import Normal
from PPO_multi.model_base import get_network_builder, device, put_on_device
MAX_LOG_STD = 2
MIN_LOG_STD = -5


# @put_on_device(device)
class Model(nn.Module):
    def __init__(self, obs_dim, actor_hidden, critic_hidden, action_dim, network):
        super().__init__()
        self.action_dim = action_dim
        if isinstance(obs_dim, int):
            actor_hidden_ = [obs_dim] + actor_hidden
            critic_hidden_ = [obs_dim] + critic_hidden
            actor = [nn.Linear(actor_hidden_[i], actor_hidden_[i+1]) for i in range(len(actor_hidden))]
            self.actor = [nn.Sequential(i, nn.ReLU()) for i in actor]
            self.actor_mean = nn.Linear(actor_hidden[-1], action_dim)
            self.actor_log_std = nn.Linear(actor_hidden[-1], action_dim)
            critic = [nn.Linear(critic_hidden_[i], critic_hidden_[i+1]) for i in range(len(critic_hidden))]
            self.critic = [nn.Sequential(i, nn.ReLU()) for i in critic]
            self.critic_v = nn.Linear(critic_hidden[-1], 1)
        else:
            self.actor = [get_network_builder(network)(),
                          nn.Sequential(
                              nn.Linear(96, 256),
                              nn.ReLU(),
                              nn.Linear(256, 128),
                              nn.ReLU(),
                              nn.Linear(128, action_dim))]
            self.critic = [get_network_builder(network)(),
                           nn.Sequential(
                              nn.Linear(96, 256),
                              nn.ReLU(),
                              nn.Linear(256, 128),
                              nn.ReLU(),
                              nn.Linear(128, 1))]

    def init(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(module.weight)

    def forward(self, cur_obs_tensor):
        actor_h = critic_h = cur_obs_tensor
        for module in self.actor:
            actor_h = module(actor_h)
        actor_mean = self.actor_mean(actor_h)
        actor_std = self.actor_log_std(actor_h).clamp(MIN_LOG_STD, MAX_LOG_STD).exp()
        for module in self.critic:
            critic_h = module(critic_h)
        critic_v = self.critic_v(critic_h)
        return actor_mean, actor_std, critic_v


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, actor_hidden, critic_hidden, action_dim, network):
        super(ActorCritic, self).__init__()
        self.model = Model(obs_dim, actor_hidden, critic_hidden, action_dim, network)

    def select_action(self, cur_obs_tensor, max_action=1.0):
        m, std, v = self.model(cur_obs_tensor)
        dist = Normal(m, std)
        action = dist.sample().clamp(-max_action, max_action)
        log_prob = dist.log_prob(action).sum(1)
        return action[0].numpy(), log_prob.detach(), v.detach().item()

    def compute_action(self, cur_obs_tensor):
        m, std, v = self.model(cur_obs_tensor)
        dist = Normal(m, std)
        entropy = dist.entropy().sum(1, keepdim=True)
        return dist, entropy, v
