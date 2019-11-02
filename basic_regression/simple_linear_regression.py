#coding=utf-8
import numpy as np

def cal_error(b,w,pts):
    # 初始化误差
    totalError = 0
    for i in range(0, len(pts)):
        x = pts[i,0] #获取 x
        y = pts[i,1] #获取 y
        # 计算所有样本点估计值到期望值间距离的平方差和
        totalError += (y - (w * x + b)) ** 2
    # 取平均值
    return totalError / float(len(pts))

# 定义每一次迭代
def step_gradient(b_current, w_current, pts, learningRate):
    b_gradient = 0 # b 变化速度
    w_gradient = 0 # w 变化速度
    # 获取样本数
    N = float(len(pts))
    for i in range(0, len(pts)):
        x = pts[i, 0]
        y = pts[i, 1]
        # 分别计算损失函数对于 b 和 w 的偏导数来作为 b 和 w 变化速度
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    # 根据导数(导数也就损失函数下降率)和学习率来决定调整参数幅度(步长)
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

def gradient_descent_runner(pts,starting_b,starting_w,learning_rate,mun_iterations):
    # 接受参数
    b = starting_b
    w = starting_w

    for i in range(mun_iterations):
        # 每一次迭代都会更新 b 和 w 
        b, w = step_gradient(b,w,np.array(pts),learning_rate)
    return [b,w]
    
    
def run():
    # 第一步 准备数据
    points = np.genfromtxt('data/data.csv',delimiter=',')

    # 第二步 定义超参数
    learning_rate = 0.0001 #更新模型的频率
    init_w = 0 #定义权重(一元线性方程斜率)
    init_b = 0 #定义偏置值(一元线性方程的截距)
    mun_iterations = 10000

    #训练模型
    print 'starting gradient descent at b = {0}, w ={1}, error ={2}'.format(init_b,init_w,cal_error(init_b,init_w,points))
    [b,w] = gradient_descent_runner(points,init_b,init_w,learning_rate, mun_iterations)
    print 'ending gradient descent at b = {0}, w ={1}, error ={2}'.format(b,w,cal_error(b,w,points))


if __name__ == '__main__':
    run()