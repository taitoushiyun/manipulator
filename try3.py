
import numpy as np
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi

a = [1, 2, 3, 4]
b = a[:]
b[3] /= 4
print(a)