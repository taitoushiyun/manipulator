import numpy as np

gate_data = [[565, 70],
             [565, 166],
             [673, 166],
             [805, 166],
             [886, 236],
             [844, 328],
             [892, 416],
             [1003, 413],
             [1000, 527],
             [1000, 624],
             [1000, 710],
             [910, 710],
             [920, 624],
             [918, 537],
             [840, 537],
             [840, 620],
             [840, 711],
             [762, 711],
             [675, 711],
             [675, 636],
             [675, 543],
             [672, 432],
             [566, 362],
             [459, 432],
             [459, 543],
             [459, 636],
             [459, 711],
             [369, 711],
             [289, 711],
             [289, 620],
             [289, 537],
             [209, 537],
             [209, 620],
             [209, 711],
             [130, 711],
             [130, 624],
             [130, 527],
             [130, 413],
             [237, 413],
             [290, 366],
             [287, 287],
             [247, 234],
             [336, 166],
             [445, 166],
             [565, 166]]

def create_path_0():
    path_list = []
    for i in range(19):
        path_list.append([1.2 + 0.2 * np.cos(i * 2 * np.pi / 18), 0.2 * np.sin(i * 2 * np.pi / 18), 1])
    return np.array(path_list)


def create_path_1():
    path_list = []
    path_list.append([1.4, 0, 1])
    for i in range(1000):
        path_list.append([1.2, 0.2 * np.sin(i * 2 * np.pi / 18), 1 + 0.2 * np.cos(i * 2 * np.pi / 18)])
    return np.array(path_list)


def create_path_2():
    path_list = np.array(gate_data)
    path_list = (path_list - np.array([565, 453])) / 1000
    path_list[:, 1] = -path_list[:, 1]
    path_list += np.array([0, 1])
    path_list = np.concatenate([np.ones((45, 1)) * 1.1, path_list], axis=-1)
    return path_list


PATH_LIST = list()
PATH_LIST.append(create_path_0())
PATH_LIST.append(create_path_1())
PATH_LIST.append(create_path_2())

if __name__ == '__main__':
    print(create_path_2())
    print(np.linalg.norm(np.array([0.2, 0, 1])-np.array([1.1, 0, 1.5]), axis=-1))
    print(np.linalg.norm(np.array([565, 453]) - np.array([1000, 711]), axis=-1))
