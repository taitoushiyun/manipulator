import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
import numpy as np
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'PPO'))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'mujoco'))
sns.set_style('whitegrid')


def smoother(x, a=0.9, w=10, mode="moving"):
    if mode is "moving":
        y = [x[0]]
        for i in range(1, len(x)):
            y.append((1 - a) * x[i] + a * y[i - 1])
    elif mode is "window":
        y = []
        for i in range(len(x)):
            y.append(np.mean(x[max(i - w, 0):i + 1]))
    else:
        raise NotImplementedError
    return y


def plot_curve(index, dfs, label=None, shaded_err=False, shaded_std=True):
    color = sns.color_palette()[index]
    N = np.min([len(df["exploration/num steps total"]) for df in dfs])
    x = dfs[0]["exploration/num steps total"][:N] / 1e6
    ys = [smoother(df["evaluation/Average Returns"][:N], w=10, mode="window") for df in dfs]
    print(len(ys), len(ys[0]), type(ys))
    y_mean = np.mean(ys, axis=0)
    y_std = np.std(ys, axis=0)
    y_stderr = y_std / np.sqrt(len(ys))
    if label is None:
        lin = plt.plot(x, y_mean, color=color)
        # plt.semilogy(x, y_mean, color=color)
    else:
        lin = plt.plot(x, y_mean, color=color, label=label)
    if shaded_err:
        plt.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=color, alpha=.4)
    if shaded_std:
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=.2)
    plt.legend(loc=4)
    return lin


if __name__ == '__main__':
    with open('/home/cq/code/manipulator/plot/goal_compare.txt', 'r') as f:
        data_2 = [[] for _ in range(2)]
        for i in range(500):
            data_temp = list(map(float, f.readline().split()))
            for j in range(2):
                data_2[j].append(data_temp[2 * j + 1])

    os.makedirs('saved_fig', exist_ok=True)
    df = pd.DataFrame(columns=('method', 'seed', 'Episodes', 'Return'))
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

    dof = [16]
    dof_name = ['r <= 0.02', 'r <= 0.01']
    for i in range(1):

        with open(f'/home/cq/.visdom/mani_{dof[i]}.json', 'r') as f:
            t = json.load(f)
        d = t['jsons']['mean rewards']['content']['data'][0]
        x = d['x'][1:501]
        y = d['y'][1:501]
        # x = smoother(x, a=0.9, w=10, mode="window")
        # y = smoother(y, a=0.9, w=10, mode="window")
        for x_, y_ in zip(x, y):
            df = df.append([{'method': dof_name[i], 'seed': 1, 'Episodes': float(x_), 'Return': float(y_)}], ignore_index=False)

    data = data_2[1]
    data = smoother(data, a=0.9, w=10, mode="window")
    for i in range(500):
        df = df.append([{'method': 'r <= 0.01', 'seed': 1, 'Episodes': float(i), 'Return': float(data[i])}],
                       ignore_index=False)

    palette = sns.color_palette("deep", 2)
    g = sns.lineplot(x='Episodes', y="Return", data=df, hue="method", style="method", dashes=False, palette=palette,
                     hue_order=dof_name)

    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/goal_effect.png', bbox_inches='tight')
    plt.show()
