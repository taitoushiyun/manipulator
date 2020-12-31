import numpy as np
from matplotlib import pyplot as plt

x = 0.1
y = 0.05
# density = 100
# ab = [np.linspace(-x, x, density),
#       np.linspace(y, y, density)]
# cd = [np.linspace(-x, x, density),
#       np.linspace(-y, -y, density)]
# ac = [np.linspace(-x, -x, density),
#       np.linspace(-y, y, density)]
# bd = [np.linspace(x, x, density),
#       np.linspace(-y, y, density)]
line = []
line.append([[-x, x], [y, y]])
line.append([[-x, x], [-y, -y]])
line.append([[-x, -x], [-y, y]])
line.append([[x, x], [-y, y]])
plt.xlim(-0.5, 1.5)
plt.ylim(-1, 1)
ax = plt.gca()
ax.set_aspect(1)

for i in range(4):
    plt.plot(*line[i], color='red')
plt.plot([0.1, 1.2], [0, 0], color='red')
plt.show()

