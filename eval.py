import time
from vrep_con.vrep_utils import ManipulatorEnv
from PPO.actor_critic import Actor_critic
import torch
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    goal_index = {'easy': [0, 20, 0, 20, 0, -10, 0, -15, 0, 20],
                  'hard': [0, 20, 0, 15, 0, 20, 0, 20, 0, 20],
                  'super hard': [0, -50, 0, -50, 0, -50, 0, -20, 0, -10]}
    env_config = {
        'distance_threshold': 0.02,
        'reward_type': 'dense',
        'max_angles_vel': 10,  # 10degree/s
        'num_joints': 10,
        'goal_set': goal_index['super hard'],
    }
    env = ManipulatorEnv(0, env_config)
    policy = actor_critic = Actor_critic(env=env,
                                         actor_obs_dims=22, actor_hidden_sizes=[64, 64],
                                         actor_action_dims=5, critic_obs_dims=22,
                                         critic_hidden_sizes=[64, 64])

    action_records = [[] for _ in range(5)]

    for i in range(210, 224):
        model = torch.load(f'PPO/checkpoints/{i}.pth')  # 'PPO/checkpoints/40.pth'
        actor_critic.load_state_dict(model)
        print(f'episode {i}')
        action_record = []
        cur_obs = env.reset()
        while True:
            action = actor_critic.eval_action(torch.FloatTensor(cur_obs[None]))
            next_obs, reward, done, info = env.step(action)
            action_record.append(cur_obs[:5])
            cur_obs = next_obs
            if done:
                action_record = np.asarray(action_record)
                for j in range(5):
                    action_records[j].append(list(action_record[:, j]))
                time.sleep(1)
                break
    # print(cur_obs[env.j_ang_idx])
    # time.sleep(1000)
    # plt.figure(figsize=(20, 4))
    # for j in range(5):
    #     plt.subplot(1, 5, j + 1)
    #     for i in range(5):
    #         plt.plot(action_records[j][i], label=f'episode {i}')
    #         plt.xlabel('stimulation steps')
    #         if j == 0:
    #             plt.ylabel('angle')
    #         plt.legend(loc='lower right')
    #         plt.title(f'joint_{j}')

    # plt.savefig('5dof.eps')
    # plt.show()
    time.sleep(1000)

    # env.end_simulation()

