import visdom
import numpy as np
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi
import time
# Y = np.random.rand(100)
# Y = (Y[Y > 0] + 1.5).astype(int),  # 100个标签1和2
# goal_list = []
# for goal_index in range(4):
#     goal = np.array([0.7 + 0.1 * np.cos(goal_index * np.pi / 4), 0.1 * np.sin(goal_index * np.pi / 4), 1])
#     print(goal)
#     goal_list.append(goal)
# goal_list = np.array(goal_list)

# viz = visdom.Visdom(port=6016, env='a_test')
# viz.scatter(
#     X=goal_list,
#     Y=np.ones((36, )),
#     opts={
#         'title': '3D Scatter',
#         'legend': ['Men', 'Women'],
#         'markersize': 5
#     }
# )
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
# print((1.4 - 0.2) // 0.01)
# print(np.random.randint(1, 10, size=(1, 3)))

# print(DEG2RAD*RAD2DEG)
# print(np.vstack([np.ones((3, )), np.ones((3,))]))
print(np.linalg.norm(np.array([1.2, 0, 1.2])-np.array([0.2, 0, 1]), axis=-1))