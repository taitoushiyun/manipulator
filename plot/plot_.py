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


def plot():
    # version = [36, 38, 40, 42, 44, 46, 49, 50, 48]
    space_type = (['plane'] * 3 + ['3D'] * 3) * 4
    gamma = ['0.95', '0.8', '0.6'] * 8
    reward_type = ['distance', 'potential', 'mix']
    joints = [f'{i} DOF' for i in range(2, 7)] + ['8 DOF', '9 DOF', '10 DOF', '12 DOF']
    alg = ['ppo', 'ppo', 'td3', 'td3']
    accuracy = ['0.02', '0.015', '0.01'] * 2
    # net = ['MLP'] * 3 + ['DenseNet'] * 3
    net = ['MLP'] * 3 + ['DenseNet'] * 3
    reset_period = ['reset every episode', 'reset every 10 episode']
    start_point = ['100 epoch', '50 epoch', '20 epoch', '0 epoch']

    for i in range(9):
        file_name = f'her_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        d = t['jsons']['eval success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        x = smoother(x, a=0.9, w=10, mode="window")
        y = smoother(y, a=0.9, w=10, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'seed': 1, 'Epoch': float(x_), 'Success Rate(%)': float(y_),
                      'joints': joints[i]})
        df = df.append(a, ignore_index=False)
    g = sns.lineplot(x='Epoch', y="Success Rate(%)", data=df, hue="joints")
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/her_random_init_5', bbox_inches='tight')
    plt.show()


def plot_random_init_10():
    df = pd.DataFrame(columns=('version', 'seed', 'Epoch', 'Success Rate(%)', 'DOF', 'with normalization',
                               'space_type', 'gamma', 'joints', 'reward_type', 'alg', 'accuracy', 'Net', 'reset period',
                               'start_point'))
    version = [36, 38, 40, 42, 44, 46, 49, 50, 48]
    space_type = (['plane'] * 3 + ['3D'] * 3) * 4
    gamma = ['0.95', '0.8', '0.6'] * 8
    reward_type = ['distance', 'potential', 'mix']
    joints = [f'{i} DOF' for i in range(2, 7)] + ['8 DOF', '9 DOF', '10 DOF', '12 DOF']
    alg = ['ppo', 'ppo', 'td3', 'td3']
    accuracy = ['0.02', '0.015', '0.01'] * 2
    # net = ['MLP'] * 3 + ['DenseNet'] * 3
    net = ['MLP'] * 3 + ['DenseNet'] * 3
    reset_period = ['reset every episode', 'reset every 10 episode']
    start_point = ['100 epoch', '50 epoch', '20 epoch', '0 epoch']

    for i in range(9):
        file_name = f'her_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        d = t['jsons']['eval success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        x = smoother(x, a=0.9, w=10, mode="window")
        y = smoother(y, a=0.9, w=10, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'seed': 1, 'Epoch': float(x_), 'Success Rate(%)': float(y_),
                      'joints': joints[i]})
        df = df.append(a, ignore_index=False)
    g = sns.lineplot(x='Epoch', y="Success Rate(%)", data=df, hue="joints")
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/her_random_init_5', bbox_inches='tight')
    plt.show()


def plot_sample_effect():
    df = pd.DataFrame(columns=('version', 'seed', 'Epoch', 'Success Rate(%)', 'sample method'))
    version = [94, 95, 96, 97]
    sample_method = ['1/x', 'uniform', 'normal(0, 1)', 'normal(0, 0.3)']

    for i in range(4):
        file_name = f'her_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        d = t['jsons']['eval success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        x = smoother(x, a=0.9, w=20, mode="window")
        y = smoother(y, a=0.9, w=20, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Epoch': float(x_), 'Success Rate(%)': float(y_),
                      'sample method': sample_method[i]})
        df = df.append(a, ignore_index=False)
    sns.lineplot(x='Epoch', y="Success Rate(%)", data=df, hue="sample method")
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/her_sample_effect', bbox_inches='tight')
    plt.show()


def plot_dense_effect():
    df = pd.DataFrame(columns=('version', 'seed', 'Epoch', 'Success Rate(%)', 'net', 'accuracy'))
    version = [27, 28, 29, 90, 93, 21, 22, 20]
    net = ['DenseNet', 'DenseNet', 'DenseNet', 'SimpleDenseNet', 'SimpleDenseNet', 'MLP', 'MLP', 'MLP']
    accuracy = ['0.02', '0.015', '0.01', '0.015', '0.01', '0.02', '0.015', '0.01']

    for i in range(8):
        file_name = f'her_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        d = t['jsons']['eval success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        x = smoother(x, a=0.9, w=10, mode="window")
        y = smoother(y, a=0.9, w=10, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Epoch': float(x_), 'Success Rate(%)': float(y_),
                      'net': net[i], 'accuracy': accuracy[i]})
        df = df.append(a, ignore_index=False)
    sns.lineplot(x='Epoch', y="Success Rate(%)", data=df, hue="accuracy", style='net')
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/her_dense_precise_effect', bbox_inches='tight')
    plt.show()


def plot_dense_effect2():
    df = pd.DataFrame(columns=('version', 'seed', 'Epoch', 'Success Rate(%)', 'net'))
    version = [ 100, 80]
    net = ['MLP', 'DenseNet']

    for i in range(len(version)):
        file_name = f'her_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        d = t['jsons']['eval success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        x = smoother(x, a=0.9, w=10, mode="window")
        y = smoother(y, a=0.9, w=10, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Epoch': float(x_), 'Success Rate(%)': float(y_),
                      'net': net[i]})
        df = df.append(a, ignore_index=False)
    sns.lineplot(x='Epoch', y="Success Rate(%)", data=df, hue="net")
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/her_dense_precise_effect_2', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    os.makedirs('saved_fig', exist_ok=True)
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    sns.set_palette(sns.color_palette('deep', 3))
    plot_dense_effect()


