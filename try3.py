import visdom
import numpy as np
import time

viz = visdom.Visdom(port=6016, env='a_test')
time_a = time.time()
viz.heatmap(
    X=np.random.randint(1, 10, (200, 120)),
    opts={
        'columnnames': list(map(lambda x: '%.2f'% x, list(np.linspace(0.2, 1.4, num=120)))),
        'rownames': list(map(lambda x: '%.2f'% x, list(np.linspace(0, 2, num=200)))),
        'colormap': 'Viridis',       # 'Electric'
    }
)
time_b = time.time()
print(time_b - time_a)
# import time
# a = time.time()
# data = list(map(lambda x: '%.2f'% x, list(np.linspace(0, 2, num=200))))
# print(len(data))
# b = time.time()
# print(b-a)
# c = ['%.2f'% 0.1234]
# print(c)
