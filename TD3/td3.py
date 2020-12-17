import torch.nn as nn
import torch.optim as optim
import torch
import visdom


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
                 code_version,
                 vis_port
                 ):
        self.env = env
        self.policy = policy
        self.target_policy = target_policy
        self.pool = pool
        self.sampler = sampler
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.tau = tau
        self.training_step = 0
        self.gamma = gamma
        self.update_freq = update_freq
        self.num_epoches = num_epoches
        self.epoch_len = epoch_len
        self.save_freq = save_freq
        self.code_version = code_version
        self.vis_port = vis_port
        self.vis = visdom.Visdom(port=self.vis_port, env=self.code_version)

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
        dones = batch_data['terminals']
        next_obs = torch.tensor(batch_data['next_observations'], dtype=torch.float)
        next_action = self.target_policy(next_obs)
        noise = torch.normal(torch.zeros_like(next_obs), self.target_policy_noise)
        noise = torch.clamp(noise, -self.target_policy_noise_clip, self.target_policy_noise_clip)
        noisy_next_actions = torch.clamp(next_action + noise,
                                         -self.policy.max_action, self.policy.max_action)
        qf1 = self.target_qf1(next_obs, noisy_next_actions)
        qf2 = self.target_qf2(next_obs, noisy_next_actions)
        v_preds = torch.min(qf1, qf2)
        target_q = rewards + (1. - dones) * self.gamma * v_preds

        qf1_loss = self.qf1_criterion(self.qf1(cur_obs, actions), target_q.detach())
        self.qf1_opt.zero_grad()
        qf1_loss.backward()
        self.qf1_opt.step()

        qf2_loss = self.qf1_criterion(self.qf2(cur_obs, actions), target_q.detach())
        self.qf2_opt.zero_grad()
        qf2_loss.backward()
        self.qf2_opt.step()

        if self.training_step % self.update_freq == 0:
            new_actions = self.policy(cur_obs)
            q_loss = -self.qf1(cur_obs, new_actions).mean()
            self.policy_opt.zero_grad()
            q_loss.backward()
            self.policy_opt.step()

            for source_param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)
            for source_param, target_param in zip(self.qf1.parameters(), self.target_qf1.paramters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)
            for source_param, target_param in zip(self.qf2.parameters(), self.target_qf2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)
        self.training_step += 1

    def eval(self):
        pass

    def sample(self):

        pass