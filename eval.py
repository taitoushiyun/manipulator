import time
from vrep_con.vrep_utils import ManipulatorEnv
from PPO.actor_critic import Actor_critic
import torch
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid')
#
# def plot(data):
#     plt.figure(figsize=(8, 6), dpi=100)
#     plt.xlabel('simulation step')
#     plt.ylabel('angle')
#     color = sns.color_palette()[0]



if __name__ == '__main__':
    env = ManipulatorEnv(0)
    policy = actor_critic = Actor_critic(env=env,
                                         actor_obs_dims=22, actor_hidden_sizes=[64, 64],
                                         actor_action_dims=5, critic_obs_dims=22,
                                         critic_hidden_sizes=[64, 64])
    model = torch.load('44.pth')
    actor_critic.load_state_dict(model)
    action_records = [[] for _ in range(5)]
    for i in range(1):
        print(f'episode {i}')
        action_record = []
        cur_obs = env.reset()
        while True:
            action, _, _ = actor_critic.select_action(torch.FloatTensor(cur_obs[None]))
            next_obs, reward, done, info = env.step(action)
            action_record.append(cur_obs[:5])
            cur_obs = next_obs
            if done:
                action_record = np.asarray(action_record)
                for j in range(5):
                    action_records[j].append(list(action_record[:, j]))
                break
    time.sleep(1000)
    # plt.figure(figsize=(20, 4))
    # for j in range(1):
    #     plt.subplot(1, 5, j + 1)
    #     for i in range(5):
    #         plt.plot(action_records[j][i], label=f'episode {i}')
    #         plt.xlabel('stimulation steps')
    #         if j == 0:
    #             plt.ylabel('angle')
    #         plt.legend(loc='lower right')
    #         plt.title(f'joint_{j}')
    #
    # plt.savefig('5dof.eps')
    # plt.show()


    # env.end_simulation()

