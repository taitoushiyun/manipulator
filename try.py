import torch.multiprocessing as mp
from vrep_con.vrep_utils import ManipulatorEnv
from vrep_con.vrep_utils import DEG2RAD
import numpy as np

def simulation(index):
    env = ManipulatorEnv(index)
    for j in range(3):
        obs = env.reset()
        action = np.random.randn(2)
        for i in range(20):
            obs, reward, done, info = env.step(action)
    env.end_simulation()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []
    for i in range(3):
        processes.append(mp.Process(target=simulation, args=(i,)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()

