import matplotlib.pyplot as plt
import numpy as np

a = 3
b = 1/2
x = 0.1
dx = 0.00001
sw = 'b**n * np.cos(a**n * np.pi * x)'
x = np.arange(-1.999, 2.099 + dx, dx)
plt.plot(x, sum(eval(sw) for n in range(0,100)))
plt.grid = True
plt.show()
