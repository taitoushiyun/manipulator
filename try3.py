
import numpy as np
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi

a = 6
for i in range(6):
    a -= np.cos(10*(i+1)*DEG2RAD)
print(a)
b = 0
for i in range(6):
    b += np.sin(10*(i+1)*DEG2RAD)
print(b)
