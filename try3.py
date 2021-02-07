
import numpy as np
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi

a = np.asarray([[1], [2], [3]])
b = np.ones((3, 3))
print(a, b)
c = a * b
print(c)