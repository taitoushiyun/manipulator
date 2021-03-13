import numpy as np
import torch

a = np.array(list(range(10)))
print(a)
b = np.absolute(a[range(0, 10, 2)] - a[range(1, 10, 2)]).sum(axis=-1)
print(b)

c = torch.tensor(list(range(10)))
print(c)
d = torch.absolute(c[range(0,10,2)]-c[range(1, 10, 2)]).sum(dim=-1)
print(d)