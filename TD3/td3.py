import torch.nn as nn
import torch.optim as optim
import torch


class TD3(object):
    def __init__(self,
                 env,
                 policy,
                 target_policy,
                 pool,
                 sampler,
                 qf1,
                 qf2,
                 target_qf1,
                 target_qf2,
                 target_policy_noise,
                 target_policy_noise_clip,
                 tau,
                 gamma,
                 update_freq,
                 num_epoches,
                 epoch_len,
                 save_freq,
                 ):
        self.env = env
        self.policy = policy
        self.pool = pool
        self.sampler = sampler
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.tau = tau
        self.training_step = 0
        self.gamma = gamma
        self.update_freq = update_freq
        self.num_epoches = num_epoches
        self.epoch_len = epoch_len
        self.save_freq = save_freq

        self.qf1_criterion = nn.MSELoss()
        self.qf1_opt = optim.Adam(self.qf1.parameters())
        self.qf2_opt = optim.Adam(self.qf2.parameters())
        self.policy_opt = optim.Adam(self.policy.paramters())

    def train(self):
        for epoch in range(self.num_epoches):
            for t in range(self.epoch_len):
                self.sampler.sample()
                if not self.sampler.batch_ready():
                    continue
                for i in range(self.epoch_len):
                    self._training(iter_step=t + epoch * self.epoch_len,
                                   )
            if epoch % self.save_freq == 0 and epoch >= 400:
                torch.save(self.policy.state_dict(), save_path)
                self.eval()

    def _training(self, batch_data):
        cur_obs = torch.tensor(batch_data['observations'], dtype=torch.float)
        actions = torch.tensor(batch_data['actions'], dtype=torch.float)
        rewards = torch.tensor(batch_data['rewards'], dtype=torch.float)
        dones = torch.tensor(batch_data['terminals'], dtype=torch.float)
        next_obs = torch.tensor(batch_data['next_observations'], dtype=torch.float)
        self.target_policy(next_obs)




    def eval(self):
        pass

    def sample(self):

        pass