# coding=utf-8

import pandas as pd
'''
假设所有女性乘客都是幸存者，然后将估计值和真实值进行对比
来检查预测的结果
'''
# 获取训练集
train = pd.read_csv("data/train.csv")
# 添加新列为
train["Hyp"] = 0
# 通过乘客性别为 female 将 Hyp 赋值 1
train.loc[train.sex == "female","Hyp"] = 1
# 验证估计值
train["Result"] = 0
train.loc[train.survived==train["Hyp"],"Result"] = 1
print train["Result"].value_counts()
print train["Result"].value_counts(normalize=True)

'''
1    701
0    190
1    0.786756
0    0.213244
'''