import visdom
import numpy as np
import time

# viz = visdom.Visdom(port=6016, env='a_test')
# time_a = time.time()
# viz.heatmap(
#     X=np.random.randint(1, 10, (200, 120)),
#     opts={
#         'columnnames': list(map(lambda x: '%.2f'% x, list(np.linspace(0.2, 1.4, num=120)))),
#         'rownames': list(map(lambda x: '%.2f'% x, list(np.linspace(0, 2, num=200)))),
#         'colormap': 'Viridis',       # 'Electric'
#     }
# )
# time_b = time.time()
# print(time_b - time_a)
# import time
# a = time.time()
# def fun():
#     return 1, 2
# data = list(map(lambda x: '%.5f'% x, list(np.linspace(0, 2, num=201, endpoint=True))))
# print(len(data))
# print(data)
# print(-1 // 0.01)
# a= np.ones((3,3))
# print(a[fun()])
# b = time.time()
# print(b-a)
# c = ['%.2f'% 0.1234]
# print(c)
print((1.4 - 0.2) // 0.01)