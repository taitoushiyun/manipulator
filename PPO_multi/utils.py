import numpy as np
import torch
import random


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def delta_time(time_):
    time_ = int(time_)
    day = time_ / 86400
    hour = time_ % 86400 / 3600
    miniute = time_ % 3600 / 60
    sec = time_ % 60
    return "%dday %.2d:%.2d:%.2d" % (day, hour, miniute, sec)


def save_model(model, filename='model.pth'):
    torch.save(
        model.state_dict(),
        filename
    )


if __name__ == "__main__":
