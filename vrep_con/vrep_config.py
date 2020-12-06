import numpy as np

#environ_cond_name = ['Joint_1', 'Gjoint_1', 'Gjoint_2']
# please check the function: _set_initial_state(self, condition) for more information.

VREP_Config = {
    'max_torque': np.ones(24)*10000000,                    # maximum torque exerted on each joint.
    'T': 20,
    'distance_threshold': 0.01,
    'buffer_type': 'HER',
    'reward_type': 'dense',
    'max_angles_vel': 10,  # 10degree/s
    'state_dim': 22,
    'num_joints': 10,
    'max_episode_steps': 50
}