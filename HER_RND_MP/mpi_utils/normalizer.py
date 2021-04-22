import threading
import numpy as np
import torch
# from mpi4py import MPI

class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # update the total stuff
        self.total_sum += v.sum(axis=0)
        self.total_sumsq += (np.square(v)).sum(axis=0)
        self.total_count += v.shape[0]
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)


class normalizer_torch:
    def __init__(self, device, size, eps=1e-2, default_clip_range=np.inf):
        self.device = device
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # get the total sum sumsq and sum count
        self.total_sum = torch.zeros(self.size, dtype=torch.float32, device=device)
        self.total_sumsq = torch.zeros(self.size, dtype=torch.float32, device=device)
        self.total_count = torch.ones(1, dtype=torch.float32, device=device)
        # get the mean and std
        self.mean = torch.zeros(self.size, dtype=torch.float32, device=device)
        self.std = torch.ones(self.size, dtype=torch.float32, device=device)

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # update the total stuff
        self.total_sum += v.sum(dim=0)
        self.total_sumsq += (torch.square(v)).sum(dim=0)
        self.total_count += v.shape[0]
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = torch.sqrt(torch.maximum(torch.square(torch.tensor(self.eps, device=self.device)), (self.total_sumsq / self.total_count) - torch.square(
            self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return torch.clip((v - self.mean) / (self.std), -clip_range, clip_range)