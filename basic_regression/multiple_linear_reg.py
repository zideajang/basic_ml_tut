# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

bh_data = load_boston()

# print(bh_data.data)
print(bh_data.data[0])
print(bh_data.data[0][5])
print(bh_data.data[0][6])
# a = np.array([0,1,2,3,4,5,6])
# print(a[2])
# ['filename', 'data', 'target', 'DESCR', 'feature_names']
# data 数据 feature_names 特征名称 和 DESCR 表示

# 目标 [[rm,age,price]]

# print(bh_data.get('data'))
# 
# print(bh_data.get('feature_names'))
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# 我们 RM 和 AGE 来组成想要数据集

data = bh_data.data
boston = pd.DataFrame(bh_data.data,columns=bh_data.feature_names)
boston['MEDV'] = bh_data.target
# print(type(bh_data.target))
# print(bh_data.target[:10])
# print(bh_data.DESCR)