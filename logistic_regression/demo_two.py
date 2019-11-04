# coding=utf-8
# 初始化 W
# 更新 W $W-= \eta \dot \frac{\partial L}{\partial W} $

import numpy as np
import matplotlib.pyplot as plt

# 数据，每一行为一个数据，数组中前两位表示特征，最后一位表示真实值，1 或 0 分别代表两个不同分类
data = np.array([[1,2,1,0],[1,2,2,0],[1,5,4,1],[1,4,5,1],[1,4,5,1],[1,2,3,0],[1,3,2,0],[1,6,5,1],[1,4,1,0],[1,6,3,1],[1,7,4,1]],dtype=np.float)
X = data[:,:-1]
y_label = data[:,-1]
y = np.vstack(y_label)

# print(np.reshape(y,(None,1)))
print(np.shape(X))
print(np.shape(y))



def w_calc(x_data,y_data, learning_rate=0.001, iter=10001):
    # 初始化 w
    W = np.mat(np.random.randn(3,1))
    # print(W)
    # 更新
    for i in range(iter):
        H = 1/(1+np.exp(-X*W))
        dw = X.T*(H-y_data) # (3,1) 
        W -= learning_rate * dw
    return W


# print(w_calc(X,y))
W = w_calc(X,y)
print(W)
w_0 = W[0,0]
w_1 = W[1,0]
w_2 = W[2,0]
print(w_0,w_1,w_2)
'''
W 求解之后画决策边界
$$ z = w_0 + w_1x_1 + w_2x_2$$
$x_1$ 对应横坐标，$x_2$ 对应纵坐标，决策边界 z = 0 因此得到方程
$$ 0 = w_0 + w_1x_1 + w_2x_2$$
$$ w_2x_2 = -w_0 - w_1x_1 $$
$$x_2 = - \frac{w_0}{w_2} - \frac{w_1}{w_2}$
'''

plotx1 = np.arange(1,7,0.01)
plotx2 = -w_0/w_2 - w_1/w_2 * plotx1
plt.plot(plotx1,plotx2,c='r',label='decision boundary')
'''
[[-6.24718321]
 [ 0.80605377]
 [ 1.14923242]]
 '''
plt.scatter(X[:,1][y_label==0],X[:,2][y_label==0],marker='^',s=150,label=0)
plt.scatter(X[:,1][y_label==1],X[:,2][y_label==1],label=1)
plt.grid()
plt.legend()

plt.show()