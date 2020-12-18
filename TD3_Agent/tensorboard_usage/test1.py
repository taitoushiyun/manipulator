from tensorboardX import SummaryWriter
import torch
import time

writer = SummaryWriter()

for i in range(0, 100, 5):
    dummy_s1 = torch.rand(1) * i
    writer.add_scalar('data/episode_reward', dummy_s1[0], i)
    writer.add_scalar('data/step', i, i)
    time.sleep(10)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()







