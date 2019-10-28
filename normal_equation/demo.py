#coding=utf-8

import numpy as np

# size/number of bedrooms/ number of floors/age of home years/price
# 第一个参数表示偏置值，这里有一些样本
X = np.array([
    [1,2104,5,1,45]
    [1,1416,3,2,40]
    [1,1534,3,2,30]
    [1,852,2,1,36]
])

# 预期值
y_ = np.array([460,232,315,178])

# 