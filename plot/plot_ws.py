import gym
import numpy as np
import visdom
from matplotlib import pyplot as plt
import matplotlib as mpl
import sys
sys.path.append('/home/cq/code/manipulator')
from vrep_con.dh_convert import DHModel
import seaborn as sns
import time
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


'''
45度均匀采样的平面样本分布
'''
time_a = time.time()
joints = 24
dh = DHModel(0, joints)
sample_list = []
num_sample = 10000
for i in range(num_sample):
    # 平面关节均匀分布采样
    # theta = np.vstack((np.zeros((joints//2,)),
    #                    45 * DEG2RAD * np.random.uniform(low=-1, high=1,
    #                                                     size=(joints//2,)))).T.flatten()
    theta = 45 * DEG2RAD * np.random.uniform(low=-1, high=1,
                                                        size=(joints,))
    # theta = 45 * DEG2RAD * (5 - 1 / np.random.uniform(low=0.2, high=1.2, size=(joints // 2,))) / (5 - (1 / 1.2)) * np.random.choice([-1, 1], size=(joints // 2, ))
    # theta = np.vstack((np.zeros((joints // 2,)),
    #                    theta)).T.flatten()


    # # 平面关节正态分布采样45 * DEG2RAD
    # theta = np.vstack((np.zeros((joints // 2,)),
    #                    0.5 * 45 * DEG2RAD * np.random.randn(joints // 2).clip(-2, 2))).T.flatten()

    # theta = np.vstack((np.zeros((joints // 2,)),
    #                    0.5 * 45 * DEG2RAD * np.random.normal(0, 0.3, size=joints // 2).clip(-2, 2))).T.flatten()

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

# plt.figure()
# plt.scatter(x=sample_list[:, 0], y=sample_list[:, 2], s=10)
# # end = dh.forward_kinematics(np.ones(20) * 20 * DEG2RAD)
# # plt.scatter(x=end[0], y=end[2], color='red')
# # plt.scatter(x=[0.3 + 0.1 * (joints // 2 - 1) * np.sqrt(2) / 2.], y=[1], color='black', s=50)
#
# x = 0.1
# y = 0.05
# line = []
# line.append([[-x, x], [1 + y, 1 + y]])
# line.append([[-x, x], [1 - y, 1 - y]])
# line.append([[-x, -x], [1 - y, 1 + y]])
# line.append([[x, x], [1 - y, 1 + y]])
# plt.xlim(-0.2, 0.3 + 0.1 * (joints // 2))
# plt.ylim(1 - 0.1 * (joints // 2), 1 + 0.1 * (joints // 2))
# ax = plt.gca()
# ax.set_aspect(1)
#
# for i in range(4):
#     plt.plot(*line[i], color='red')
# plt.plot([0.1, 0.2 + 0.1 * (joints // 2)], [1, 1], color='red')
# plt.xlabel('X axis')
# plt.ylabel('Z axis')
# plt.savefig('saved_fig/cws_6_all.png', bbox_inches='tight')
# plt.show()
#
# time_b = time.time()
# print(time_b - time_a)
# print(dh.forward_kinematics(np.zeros(joints)))


# heat_map = np.zeros((301, 201))
#
# def clamp(n, minn, maxn):
#     return max(min(maxn, n), minn)
#
# for i in range(num_sample):
#     index_z = clamp((sample_list[i][2] + 0.5) // 0.01, 0, 300)
#     index_x = clamp((sample_list[i][0] + 0.5) // 0.01, 0, 200)
#     heat_map[(int(index_z), int(index_x))] = heat_map[(int(index_z), int(index_x))] + 1
#
# # ax = plt.gca()
# # ax.set_aspect(1)
# # sns.set()
# # sns.set_theme()
# # sns.heatmap(heat_map, ax=ax, cmap="YlGnBu")
# # plt.show()
# vis = visdom.Visdom(port=6016, env='a_test')
# vis.heatmap(
#     X=heat_map,
#     # win='heat map',
#     opts={
#         'Xlabel': 'X',
#         'Ylabel': 'Y',
#         'title': 'sample probability',
#         'columnnames': list(map(lambda x: '%.2f'% x, list(np.linspace(-0.5, 1.5, num=201, endpoint=True)))),
#         'rownames': list(map(lambda x: '%.2f'% x, list(np.linspace(-0.5, 2.5, num=301, endpoint=True)))),
#         'colormap': 'Viridis',       # 'Electric'
#     }
# )

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(sample_list[:, 0], sample_list[:, 1], sample_list[:, 2], label='goal path')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
set_axes_equal(ax)
ax.legend()
plt.show()