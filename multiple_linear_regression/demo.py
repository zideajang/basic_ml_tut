# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generate_dataset(n):
    x = []
    y = []
    rand_x1 = np.random.rand()
    rand_x2 = np.random.rand()

    for i in range(n):
        x1 = i # i 为顺序函数
        x2 = i/2 + np.random.rand()*n
        x.append([1,x1,x2])
        y.append(rand_x1 * x1 + rand_x2 * x2 + 1)
    return np.array(x), np.array(y)

x,y = generate_dataset(200)
# print x,y

# 创建线性模型来拟合这些数据
mpl.rcParams['legend.fontsize'] = 12

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x[:,1],x[:,2],y,label='y',s=5)
ax.legend()
ax.view_init(45,0)

plt.show()

x1 = np.linspace(-5,5,20)
x2 = np.linspace(-5,5,20)

np.random.seed(1)
# 标注化分布的
y = 2*x -3 * np.random.normal(size=x.shape)

# 创建
data = pd.DataFrame({'x1':x1,'x2':x2,'y':y})
from statsmodels
# 
model = ols()