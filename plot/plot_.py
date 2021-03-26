import os

import matplotlib.pyplot as plt
import matplotlib as mpl
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
    sample_method = ['U shape', 'Uniform', 'Normal_1', 'Normal_2']

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


def plot_gamma_effect(ylabel='Return'):
    df = pd.DataFrame(columns=('version', 'seed', 'Episodes', ylabel, 'gamma', 'space_type'))
    # version = [77, 78, 79, 80, 81, 82]  # ppo 24 joints distance
    version = [89, 90, 91, 92, 93, 94]  # ppo 12 joints distance
    # version = list(range(140, 140+6))  # td3 12 joints distance
    # version = list(range(146, 146+6))  # td3 12 joints potential
    # version = list(range(152, 158)) + [186, 185]  # td3 24 joints distance
    # version = list(range(158, 164))  # td3 24 joints potential
    gamma = ['0.95', '0.8', '0.6'] * 2 + ['0', '0']
    space_type = ['plane'] * 3 + ['3D'] * 3 + ['plane', '3D']
    for i in range(len(version)):
        file_name = f'mani_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        if ylabel == 'Return':
            d = t['jsons']['mean reward']['content']['data'][0]
        else:
            d = t['jsons']['success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        # x = smoother(x, a=0.9, w=10, mode="window")
        # y = smoother(y, a=0.9, w=10, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Episodes': float(x_), ylabel: float(y_),
                      'gamma': gamma[i], 'space_type': space_type[i]})
        df = df.append(a, ignore_index=False)

    sns.lineplot(x='Episodes', y=ylabel, data=df, hue="gamma", style='space_type')
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/ppo_6_distance.png', bbox_inches='tight')
    plt.show()


def plot_reward_effect(ylabel='Return'):
    df = pd.DataFrame(columns=('version', 'seed', 'Episodes', ylabel, 'reward_type', 'space_type'))
    version = [153, 156, 159, 162]  # td3 gamma=0.8
    # version = [154, 157, 160, 163]   # td3  gamma = 0.6
    # version = [78, 81, 84, 87]  # ppo gamma=0.8
    # version = [79, 82, 85, 88]  # ppo gamma=0.6
    reward_type = ['distance'] * 2 + ['potential'] * 2
    space_type = ['plane', '3D'] * 2
    for i in range(len(version)):
        file_name = f'td3_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        if ylabel == 'Return':
            d = t['jsons']['mean reward']['content']['data'][0]
        else:
            d = t['jsons']['success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        # x = smoother(x, a=0.9, w=10, mode="window")
        # y = smoother(y, a=0.9, w=10, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Episodes': float(x_), ylabel: float(y_),
                      'reward_type': reward_type[i], 'space_type': space_type[i]})
        df = df.append(a, ignore_index=False)

    sns.lineplot(x='Episodes', y=ylabel, data=df, hue="space_type", style='reward_type')
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/td3_reward_effect.png', bbox_inches='tight')
    plt.show()

def plot_reward_effect2(ylabel='Return'):
    df = pd.DataFrame(columns=('version', 'seed', 'Episodes', ylabel, 'reward_type', 'space_type'))
    # version = [153, 156, 159, 162]  # gamma=0.8
    version = [165, 167, 168, 169, 170]   # gamma = 0.6
    reward_type = ['distance', 'potential', 'mix', 'l2', 'l4']
    space_type = ['plane', '3D'] * 2
    for i in range(len(version)):
        file_name = f'td3_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        if ylabel == 'Return':
            d = t['jsons']['mean reward']['content']['data'][0]
        else:
            d = t['jsons']['eval success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        x = smoother(x, a=0.9, w=10, mode="window")
        y = smoother(y, a=0.9, w=10, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Episodes': float(x_), ylabel: float(y_),
                      'reward_type': reward_type[i]})
        df = df.append(a, ignore_index=False)

    sns.lineplot(x='Episodes', y=ylabel, data=df, hue="reward_type")
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/td3_reward_effect2.png', bbox_inches='tight')
    plt.show()


def plot_td3_random(ylabel='Return'):
    df = pd.DataFrame(columns=('version', 'seed', 'Episodes', ylabel, 'DOF'))
    # version = [69, 101]
    # version = [101, 107]
    version = [203, 69]
    dof = ['24 DOF', '12 DOF']
    for i in range(len(version)):
        file_name = f'td3_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        if ylabel == 'Return':
            d = t['jsons']['mean reward']['content']['data'][0]
        else:
            d = t['jsons']['success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        # x = smoother(x, a=0.9, w=20, mode="window")
        # y = smoother(y, a=0.9, w=20, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Episodes': float(x_), ylabel: float(y_),
                      'DOF': dof[i]})
        df = df.append(a, ignore_index=False)

    # y_mean = np.mean(ys, axis=0)
    # y_std = np.std(ys, axis=0)
    # y_stderr = y_std / np.sqrt(len(ys))
    # if label is None:
    #     lin = plt.plot(x, y_mean, color=color)
    #     # plt.semilogy(x, y_mean, color=color)
    # else:
    #     lin = plt.plot(x, y_mean, color=color, label=label)
    # if shaded_err:
    #     plt.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=color, alpha=.4)
    # if shaded_std:
    #     plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=.2)

    sns.lineplot(x='Episodes', y=ylabel, data=df, hue='DOF')
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/td3_random_6_12_2.png', bbox_inches='tight')
    plt.show()

def plot_td3_random_precise(ylabel='Return'):
    df = pd.DataFrame(columns=('version', 'seed', 'Episodes', ylabel, 'accuracy'))
    version = [101, 105, 106]
    accuracy = ['0.02', '0.01', '0.005']
    for i in range(len(version)):
        file_name = f'td3_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        if ylabel == 'Return':
            d = t['jsons']['mean reward']['content']['data'][0]
        else:
            d = t['jsons']['success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        x = smoother(x, a=0.98, w=100, mode="window")
        y = smoother(y, a=0.98, w=100, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Episodes': float(x_), ylabel: float(y_),
                      'accuracy': accuracy[i]})
        df = df.append(a, ignore_index=False)

    sns.lineplot(x='Episodes', y=ylabel, data=df, hue='accuracy')
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/td3_random_precise_6.png', bbox_inches='tight')
    plt.show()

def plot_test(ylabel='Return'):
    # df = pd.DataFrame(columns=('version', 'seed', 'Episodes', ylabel, 'accuracy'))

    file_name = f'test_td3'
    with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
        t = json.load(f)
    x = t['jsons']['window_396e19a570ed40']['content']['data'][1]['x']
    y = t['jsons']['window_396e19a570ed40']['content']['data'][1]['y']
    z = t['jsons']['window_396e19a570ed40']['content']['data'][1]['z']
    data1 = np.array([x, y, z]).T
    x = t['jsons']['window_396e19a570ed40']['content']['data'][0]['x']
    y = t['jsons']['window_396e19a570ed40']['content']['data'][0]['y']
    z = t['jsons']['window_396e19a570ed40']['content']['data'][0]['z']
    data2 = np.array([x, y, z]).T

    # a = []
    # for x_, y_ in zip(x, y):
    #     a.append({'version': str(version[i]), 'Episodes': float(x_), ylabel: float(y_),
    #               'accuracy': accuracy[i]})
    # df = df.append(a, ignore_index=False)
    #
    # sns.lineplot(x='Episodes', y=ylabel, data=df, hue='accuracy')
    # plt.tight_layout()
    # plt.legend(fontsize=10, loc='lower right')
    # # plt.savefig('saved_fig/test.png', bbox_inches='tight')
    # plt.show()

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], label='success')
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], label='fail')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.legend()
    plt.savefig('saved_fig/td3_eval_3.png', bbox_inches='tight')
    plt.show()


def plot_her(ylabel='Success Rate(%)'):
    df = pd.DataFrame(columns=('version', 'seed', 'Epochs', ylabel, 'DOF'))
    version = [6, 5]
    dof = ['24 DOF', '12 DOF']
    for i in range(len(version)):
        file_name = f'her_{version[i]}'
        with open(f'/home/cq/.visdom/{file_name}.json', 'r') as f:
            t = json.load(f)
        d = t['jsons']['eval success rate']['content']['data'][0]
        x = d['x']
        y = d['y']
        # x = smoother(x, a=0.98, w=100, mode="window")
        # y = smoother(y, a=0.98, w=100, mode="window")
        a = []
        for x_, y_ in zip(x, y):
            a.append({'version': str(version[i]), 'Epochs': float(x_), ylabel: float(y_),
                      'DOF': dof[i]})
        df = df.append(a, ignore_index=False)

    sns.lineplot(x='Epochs', y=ylabel, data=df, hue='DOF')
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/her_6_12.png', bbox_inches='tight')
    plt.show()

def plot_random_init():
    df = pd.DataFrame(columns=('version', 'seed', 'Epoch', 'Success Rate(%)', 'DOF', 'with normalization',
                               'space_type', 'gamma', 'joints', 'reward_type', 'alg', 'accuracy', 'Net', 'reset period',
                               'start_point'))
    # version = [50, 51]  # 10 DOF 2D
    version = [30, 98]  # 12DOF 3D
    reset_period = ['never reset', 'reset every 10 episode']


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
            a.append({'version': str(version[i]), 'seed': 1, 'Epoch': float(x_), 'Success Rate(%)': float(y_),
                      'reset_period': reset_period[i]})
        df = df.append(a, ignore_index=False)
    g = sns.lineplot(x='Epoch', y="Success Rate(%)", data=df, hue="reset_period")
    plt.tight_layout()
    plt.legend(fontsize=10, loc='upper right')
    plt.savefig('saved_fig/her_random_init_6', bbox_inches='tight')
    plt.show()


def plot_random_init_compare():
    df = pd.DataFrame(columns=('version', 'seed', 'Epoch', 'Success Rate(%)', 'DOF', 'with normalization',
                               'space_type', 'gamma', 'joints', 'reward_type', 'alg', 'accuracy', 'Net', 'reset period',
                               'start_point'))
    version = [ 80, 98, 100, 32]
    reset_period = ['DenseNet + variable reset period',
                    'DenseNet + reset every 10 episode',
                    'MLP + variable reset period',
                    'DenseNet = reset every episode']


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
            a.append({'version': str(version[i]), 'seed': 1, 'Epoch': float(x_), 'Success Rate(%)': float(y_),
                      'reset_period': reset_period[i]})
        df = df.append(a, ignore_index=False)
    g = sns.lineplot(x='Epoch', y="Success Rate(%)", data=df, hue="reset_period")
    plt.tight_layout()
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('saved_fig/her_random_init_7', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    os.makedirs('saved_fig', exist_ok=True)
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    sns.set_palette(sns.color_palette('deep', 4))
    # plot_her()   # ylabel='Success Rate(%)'
    # plot_td3_random_precise(ylabel='Success Rate(%)')
    plot_random_init_compare()


