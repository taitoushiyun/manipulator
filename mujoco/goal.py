import numpy as np


def create_path_0():
    path_list = []
    for i in range(19):
        path_list.append([1.2 + 0.2 * np.cos(i * 2 * np.pi / 18), 0.2 * np.sin(i * 2 * np.pi / 18), 1])
    return np.array(path_list)


def create_path_1():
    path_list = []
    path_list.append([1.4, 0, 1])
    for i in range(19):
        path_list.append([1.2, 0.2 * np.sin(i * 2 * np.pi / 18), 1 + 0.2 * np.cos(i * 2 * np.pi / 18)])
    return np.array(path_list)


PATH_LIST = list()
PATH_LIST.append(create_path_0())
PATH_LIST.append(create_path_1())
