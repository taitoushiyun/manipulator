import numpy as np
num_joints = 12
num_segments = 2
theta = 45 *  np.random.uniform(-1, 1, size=(2 * num_segments, ))
theta = np.vstack([np.hstack([theta[i_] * np.ones(num_joints // (2 * num_segments))
                              for i_ in range(num_segments)]),
                   np.hstack([theta[i_ + num_segments] * np.ones(num_joints // (2 * num_segments))
                              for i_ in range(num_segments)])
                   ]).T.flatten()
k = [1, 2, 3, 4, 5, 6]
print(k[[1, 3, 5]])