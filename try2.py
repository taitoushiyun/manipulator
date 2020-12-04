import numpy as np
def transfer(x):
    a = np.asarray(x)
    s = a.shape
    a = np.reshape(a, (a[0]*a[1], *a[2:]))
map(np.asarray, zip(*[[1, 2, 3]]))