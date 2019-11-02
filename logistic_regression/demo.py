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
sigma = 1
beta_x, beta_z