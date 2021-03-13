import numpy as np
import json
import matplotlib.pyplot as plt
import time
import imageio
import os
from _collections import deque

def create_gif(image_list, gif_name, duration=1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

png = []
os.makedirs('cache',  exist_ok=True)
with open('/home/cq/.visdom/her_17.json', 'r') as f:
    t = json.load(f)
    old_data = np.zeros_like(np.array(t['jsons']['epoch184']['content']['data'][0]['z']))
    queue = deque(maxlen=8)
    i = 0
    for key in t['jsons'].keys():
        if key.startswith('epoch'):
            i += 1
            print(i)
            data = t['jsons'][key]['content']['data'][0]['z']
            data = np.array(data)
            queue.append(data)
            new_data = old_data.copy()
            for data_ in queue:
                new_data += data_
            # time_a = time.time()
            plt.figure()
            plt.title(key)
            ax = plt.gca()
            ax.set_aspect(1)
            plt.imshow(new_data, cmap=plt.cm.hot)
            plt.colorbar()
            plt.savefig(f'cache/{key}.png')
            plt.close()
            png.append(f'cache/{key}.png')
            # time_b = time.time()
            # print(time_b - time_a)
    # for key, value in t['jsons']['epoch184']['content']['data'][0].items():
    #     print(key, value)
create_gif(png, 'mix8_3.gif', duration=0.3)
# x = np.random.rand(100).reshape(10, 10)
# plt.imshow(x, cmap=plt.cm.hot)
# plt.colorbar()
# plt.show()