import numpy as np
a, b, c = map(np.asarray, zip(*[[1, 2, 3, [1,2]], [4, 5, 6, [1,2, 3]], [7, 8, 9, [1, 2, 2]]])[:-1])
print(a)
print(b)
print(c)