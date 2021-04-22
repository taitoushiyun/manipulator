import numpy as np
import torch
import torch.nn as nn

class PopArt(nn.Module):
    def __init__(self, args, epsilon=1e-2, stable_rate=0.005, min_steps=100000):
        super(PopArt, self).__init__()
        if args.cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.mu = self.mu_new = torch.tensor(0.0, requires_grad=False, device=device)
        self.sigma = self.sigma_new = torch.tensor(0.0, requires_grad=False, device=device)
        self.nu = torch.tensor(0.0, requires_grad=False, device=device)
        self.beta = args.beta
        self.cnt = 1
        self.pop_is_active = False
        self.beta = args.beta
        self.epsilon = torch.tensor(epsilon, requires_grad=False, device=device)
        self.stable_rate = torch.tensor(stable_rate, requires_grad=False, device=device)
        self.min_steps = min_steps

    def update(self, target_q, critic):
        sample_size = target_q.shape[0]
        beta = min((self.beta / (1 - (1 - self.beta)**self.cnt)) * sample_size, 1)
        def update_avg(old_avg, new_sample, beta):
            new_avg = (1 - beta) * old_avg + beta * new_sample
            return new_avg
        self.mu_new = update_avg(self.mu, target_q.mean(), beta)
        self.nu_new = update_avg(self.nu, torch.square(target_q).mean(), beta)
        self.cnt += sample_size
        self.sigma_old = torch.sqrt(self.nu - torch.square(self.mu))
        self.sigma_new = torch.sqrt(self.nu_new - torch.square(self.mu_new))
        self.sigma_old = self.sigma_old if self.sigma_old > 0 else self.sigma_new

        std_is_stable = (self.cnt > self.min_steps and torch.abs(1 - self.sigma_old / self.sigma_new) < self.stable_rate)
        # if abs(self.stable_rate.item() - 0.05) < 0.0001:
        #     # print(torch.abs(1 - self.sigma_old / self.sigma_new))
        #     print(self.pop_is_active, std_is_stable, self.cnt > self.min_steps,
        #           torch.abs(1 - self.sigma_old / self.sigma_new), self.sigma)
        # print(self.pop_is_active, std_is_stable, self.cnt > self.min_steps, torch.abs(1 - self.sigma_old / self.sigma_new), )

        self.cnt_enough = self.cnt > self.min_steps
        if self.pop_is_active or std_is_stable:
            relative_sigma = self.sigma_old / self.sigma_new
            for critic_ in critic:
                critic_.upper_layer.fc.weight.data.mul_(relative_sigma)
                critic_.upper_layer.fc.bias.data.mul_(relative_sigma).add_((self.mu - self.mu_new) / self.sigma_new)
            self.pop_is_active = True
        self.mu = self.mu_new
        self.nu = self.nu_new
        self.sigma = self.sigma_new


