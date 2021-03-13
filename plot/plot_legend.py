from matplotlib import pyplot as plt
import numpy as np
import sys
def avg(a, n, mode):
    return np.convolve(a, np.ones((n,))/n, mode=mode)
# num_f = 4
# data = []
# for i in range(num_f):
#     with open('data_4.txt', 'r') as f:
#         data_temp = f.readlines()
#         def f(x):
#             return float(x.strip())
#         data.append(list(map(f, data_temp)))
num_f = 2
mean_0 = []
mean_1 = []
mean_2 = []
with open('data_0.txt', 'r') as f:
    data_0 = [[] for _ in range(num_f)]
    for i in range(500):
        data_temp = list(map(float, f.readline().split()))
        for j in range(num_f):
            data_0[j].append(data_temp[2*j + 1])

with open('data_1.txt', 'r') as f:
    data_1 = [[] for _ in range(num_f)]
    for i in range(500):
        data_temp = list(map(float, f.readline().split()))
        for j in range(num_f):
            data_1[j].append(data_temp[2*j + 1])

with open('data_2.txt', 'r') as f:
    data_2 = [[] for _ in range(num_f)]
    for i in range(500):
        data_temp = list(map(float, f.readline().split()))
        for j in range(num_f):
            data_2[j].append(data_temp[2*j + 1])


from collections import deque
for i in range(num_f):
    temp = deque(maxlen=10)
    mean_data = []
    for j in range(500):
        temp.append(data_0[i][j])
        mean_data.append(sum(temp)/len(temp))
    mean_0.append(mean_data)

for i in range(num_f):
    temp = deque(maxlen=10)
    mean_data = []
    for j in range(500):
        temp.append(data_1[i][j])
        mean_data.append(sum(temp)/len(temp))
    mean_1.append(mean_data)

for i in range(num_f):
    temp = deque(maxlen=10)
    mean_data = []
    for j in range(500):
        temp.append(data_2[i][j])
        mean_data.append(sum(temp)/len(temp))
    mean_2.append(mean_data)


plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.title('2 DOF')
plt.plot(mean_0[0], label='r <= 0.02')
plt.plot(mean_0[1], label='r <= 0.01')
plt.xlabel('episodes')
plt.ylabel('mean rewards')
plt.legend(loc='lower right')

plt.subplot(132)
plt.title('3 DOF')
plt.plot(mean_1[1], label='r <= 0.02')
plt.plot(mean_1[0], label='r <= 0.01')
plt.xlabel('episodes')
plt.ylabel('mean rewards')
plt.legend(loc='lower right')


plt.subplot(133)
plt.title('5 DOF')
plt.plot(mean_2[0], label='r <= 0.02')
plt.plot(mean_2[1], label='r <= 0.01')
plt.xlabel('episodes')
plt.ylabel('mean rewards')
plt.legend(loc='lower right')

plt.savefig('done_compare.pdf')
plt.show()


