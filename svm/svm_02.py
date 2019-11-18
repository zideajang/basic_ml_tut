#coding=utf-8
import numpy as np
from matplotlib import pyplot as plt

# 准备数据
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1,6,-1],
    [2,4,-1],
    [6,2,-1],
])

y = np.array([-1,-1,1,1,1])

for d, sample in enumerate(X):
    if d < 2:
        plt.scatter(sample[0],sample[1],s=120,marker='_',linewidths=2)
    else:
        plt.scatter(sample[0],sample[1],s=120,marker='+',linewidths=2)
plt.plot([-2,6],[6,0.5])
plt.show()

# 定义损失函数和目标函数
# Hinge loss(损失函数)是一种在 SVM 常用的损失函数
'''
$$c(x,y,f(x)) = (1 - y * f(x))_{+}$$
- x 是样本
- y 是真实值
- f(x) 是估计值
$$ \begin$$
'''
