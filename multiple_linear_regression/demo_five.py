# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt(r"data/data.csv",delimiter=',')
# print(data)

# 切分数据
x_data = data[:,:-1]
y_data = data[:,-1]

# 定义学习率
learning_rate = 0.0001
# 定义参数
w_0 = 0
w_1 = 0
w_2 = 0

number_iteration = 1000

def computer_error(w_0,w_1,w_2,x_data,y_data):
    # 初始化代价值
    totalError = 0
    for i in range(0,len(x_data)):
        # $$ y - (w_0 + w_1x^i_0 + w_2 x^i_1) $$
        totalError += (y_data[i] - (w_0 + w_1*x_data[i,0] + w_2*x_data[i,1])) ** 2
        # 求其平均值
    return totalError/float(len(x_data))


def gradient_descent_runner(x_data,y_data,w_0,w_1,w_2,learning_rate,number_iteration):
    # 计算总数据量
    m = float(len(x_data))
    # 循环 iteration 次数
    for i in range(number_iteration):
        w_0_grad = 0
        w_1_grad = 0
        w_2_grad = 0
        # 计算梯度的总和再求平均
        for j in range(0,len(x_data)):
            w_0_grad += -(1/m) * (y_data[j] - (w_1*x_data[j,0] + w_2*x_data[j,1] + w_0))
            w_1_grad += -(1/m) * x_data[j,0] * (y_data[j] - (w_1*x_data[j,0] + w_2*x_data[j,1] + w_0))
            w_2_grad += -(1/m) * x_data[j,1] * (y_data[j] - (w_1*x_data[j,0] + w_2*x_data[j,1] + w_0))
        # 更新参数
        w_0 = w_0 - (learning_rate*w_0_grad)
        w_1 = w_1 - (learning_rate*w_1_grad)
        w_2 = w_2 - (learning_rate*w_2_grad)
    return w_0,w_1,w_2

print("Starting w_0 = {0},w_1 = {1},w_2={2}, error={3}".format(w_0,w_1,w_2,computer_error(w_0,w_1,w_2,x_data,y_data)))
print("Running...")
w_0,w_1,w_2 = gradient_descent_runner(x_data,y_data,w_0,w_1,w_2,learning_rate,number_iteration)
print("After {0} w_0 = {1},w_1 = {2},w_2={3}, error={4}".format( number_iteration, w_0,w_1,w_2,computer_error(w_0,w_1,w_2,x_data,y_data)))


ax = plt.figure().add_subplot(111,projection='3d')
ax.scatter(x_data[:,0],x_data[:,1],y_data,c='r',marker='o',s=100)
x0 = x_data[:,0]
x1 = x_data[:,1]
x0,x1 = np.meshgrid(x0,x1)
z = w_0 + x0*w_1 + x1*w_2
ax.plot_surface(x0,x1,z)
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')

plt.show()