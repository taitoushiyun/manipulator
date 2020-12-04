import visdom
import numpy as np
import time


def plot(args, iter_step, reward, path_len):
    vis = visdom.Visdom(port=args.vis_port, env=args.code_version)
    vis.line(X=np.array([0]),
             Y=np.array([0]),
             win='mean rewards to time',
             opts=dict(xlabel='minutes',
                       ylabel='mean reward',
                       title='mean reward'))
    vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        win="mean path len to time",
        opts=dict(
            xlabel='minutes',
            ylabel='mean path len',
            title='mean path len'))
    vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        win="iter steps to time",
        opts=dict(
            xlabel='minutes',
            ylabel='iter step',
            title='time & iter steps'))
    time_minutes = 0
    while True:
        time.sleep(args.plot_interval)
        time_minutes += args.plot_interval / 60.
        # print(f'time is {time_minutes}')
        reward_mean = sum(reward) / len(reward) if len(reward) else 0
        path_len_mean = sum(path_len) / len(path_len) if len(path_len) else 0
        vis.line(
            X=np.array([time_minutes]),
            Y=np.array([reward_mean]),
            win='mean rewards to time',
            update='append')
        vis.line(
            X=np.array([time_minutes]),
            Y=np.array([path_len_mean]),
            win="mean path len to time",
            update='append')
        vis.line(
            X=np.array([time_minutes]),
            Y=np.array([iter_step.value]),
            win="iter steps to time",
            update='append')


