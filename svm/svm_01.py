# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

# y = 1/2x + 1
# x_data = np.random
x = np.arange(-10,10,1)
# plt.plot([-2,0],[0,1])
plt.plot(x, (0.5 * x  + 1),label="y = 0.5x + 1")
plt.xlim([-10,10])
plt.ylim([-5,20])
plt.legend()
plt.show()