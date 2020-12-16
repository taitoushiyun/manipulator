import torch.nn as nn
import torch.optim as optim
import torch


class TD3(object):
    def __init__(self,
                 env,
                 policy,
                 pool,
                 qf1,
                 qf2,
                 target_policy_noise,
                 target_policy_noise_clip,
                 tau,
                 gamma,
                 update_freq,

                 ):
        self.env = env
        self.policy = policy
        self.pool = pool
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.tau = tau
        self.training_step = 0
        self.gamma = gamma
        self.update_freq = update_freq

        self.qf1_criterion = nn.MSELoss()
        self.qf1_opt = optim.Adam(self.qf1.parameters())
        self.qf2_opt = optim.Adam(self.qf2.parameters())
        self.policy_opt = optim.Adam(self.policy.paramters())
        self.qf1_target = self.qf1.copy()
        self.qf2_target = self.qf2.copy()
        self.policy_target = self.policy.copy()

    def train(self):
        pass

    def eval(self):
        pass

    def sample(self):
        pass