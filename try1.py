import gym
import numpy as np

from matplotlib import pyplot as plt
from vrep_con.dh_convert import DHModel
import time
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi

'''
45度均匀采样的平面样本分布
'''
time_a = time.time()
joints = 24
dh = DHModel(0, joints)
sample_list = []
for i in range(5000):
    # 平面关节均匀分布采样
    theta = np.vstack((np.zeros((joints//2,)),
                       45 * DEG2RAD * np.random.uniform(low=-1, high=1,
                                                        size=(joints//2,)))).T.flatten()

    # 平面关节正态分布采样
    # theta = np.vstack((np.zeros((joints // 2,)),
    #                    0.5 * 45 * DEG2RAD * np.random.randn(joints // 2).clip(-2, 2))).T.flatten()

    # # 平面关节常曲率均匀分布采样
    # theta = 45 * DEG2RAD * np.random.uniform(-1, 1, size=(2, ))
    # theta = np.vstack((np.zeros((joints // 2,)),
    #                    np.hstack((theta[0] * np.ones(joints // 4), theta[1] * np.ones(joints // 4))))).T.flatten()

    # # 平面关节常曲率正态分布采样
    # theta = 0.5 * 45 * DEG2RAD * np.random.randn(2).clip(-2, 2)
    # theta = np.vstack((np.zeros((joints // 2,)),
    #                    np.hstack((theta[0] * np.ones(joints // 4), theta[1] * np.ones(joints // 4))))).T.flatten()

    sample = dh.forward_kinematics(theta)
    sample_list.append(sample)

sample_list = np.asarray(sample_list)

plt.figure()
plt.scatter(x=sample_list[:, 0], y=sample_list[:, 2], s=10)
# end = dh.forward_kinematics(np.ones(20) * 20 * DEG2RAD)
# plt.scatter(x=end[0], y=end[2], color='red')
plt.scatter(x=[0.3 + 0.1 * (joints // 2 - 1) * np.sqrt(2) / 2.], y=[1], color='black', s=50)

x = 0.1
y = 0.05
line = []
line.append([[-x, x], [1 + y, 1 + y]])
line.append([[-x, x], [1 - y, 1 - y]])
line.append([[-x, -x], [1 - y, 1 + y]])
line.append([[x, x], [1 - y, 1 + y]])
plt.xlim(-0.5, 0.3 + 0.1 * (joints // 2))
plt.ylim(1 - 0.1 * (joints // 2), 1 + 0.1 * (joints // 2))
ax = plt.gca()
ax.set_aspect(1)

for i in range(4):
    plt.plot(*line[i], color='red')
plt.plot([0.1, 0.2 + 0.1 * (joints // 2)], [1, 1], color='red')

plt.show()
time_b = time.time()
print(time_b - time_a)
# print(dh.forward_kinematics(np.zeros(joints)))


