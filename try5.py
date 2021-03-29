import seaborn as sns

import time
import gym
import mujoco_py
import torch


def plot_joint_sample_dist():
    import numpy as np
    import matplotlib.pyplot as plt
    list_ = [0] * 200
    for i in range(1000000):
        a = (5 - 1 / np.random.uniform(low=0.2, high=1.2, size=())) / (5 - (1 / 1.2)) * np.random.choice([-1, 1])
        # a = 0.5 * np.random.normal(0, 0.3, size=()).clip(-2, 2)
        # a = np.random.uniform(-1, 1)
        list_[int((a+1)//0.01)] += 1
    list_ = np.array(list_) / 1000000
    plt.plot(np.linspace(-1, 1, 200), list_)
    # plt.plot([-1, 1], [0.5, 0.5])
    plt.title('joint angle distribution')
    plt.xlabel('theta')
    plt.ylabel('probability')
    plt.savefig('/home/cq/code/manipulator/plot/saved_fig/curve_2', bbox_inches='tight')
    plt.show()
def visdom_speed_test():
    import visdom
    vis = visdom.Visdom(port=6016, env='a_test_3')
    vis.line(X=[0], Y=[0], win='ha')
    vis.line(X=[0], Y=[0], win='hah')
    for i in range(10000):
        vis.line(X=[i], Y=[i], win='ha', update='append')
        vis.line(X=[i], Y=[i], win='hah', update='append')
        time.sleep(10)


if __name__ == '__main__':
    print(5 * 3.14 / 180)
