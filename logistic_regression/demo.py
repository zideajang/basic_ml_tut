#coding=utf-8
# matrix math
import numpy as np
import pandas as pd
# matrix data structure
from patsy import dmatrices
# from error logging
import warnings


def sigmoid(x):
    return 1/(1+np.exp(-x))

# 准备数据生成可以预测的随机数据
np.random.seed(0)
# 阈值
tol=1e-8
# 选择 L2 正则
lam = None
max_iter = 20

r = 0.95
n = 1000
sigma = 1 #添加一些噪音数据，控制数据离散程度

beta_x, beta_z, beta_v = -4, .9, 1 
var_x, var_z, var_v = 1, 1, 4 #输入的变量

formula = 'y - x + z + v + np.exp(x) + I(v**2 + z)'

# 准备数据
# 将 x 和 z 关联在一起(身高和体重)
x, z = np.random.multivariate_normal([0,0],[[var_x,r],[r,var_z]],n).T
# 定义血压
v = np.random.normal(0,var_v,n) ** 3

A = pd.DataFrame({'x':x, 'z':z,'v':v})

A['log_odds'] = sigmoid(A[['x','z','v']].dot([beta_x,beta_z,beta_v]) + sigma * np.random.normal(0,1,n))

A['y'] = [np.random.binomial(1,p) for p in A.log_odds]
print(dmatrices)
y,X = dmatrices(formula, A, return_type='dataframe')

X.head()