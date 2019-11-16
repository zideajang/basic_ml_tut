# coding=utf-8

# 导入依赖
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint
#  加载并准备数据
names = ["sepal_length","sepal_width","petal_length","petal_width","label"]
df = pd.read_csv("data/iris.csv",header=None,names=names)

# print df.head()
# 最后列为我们分类标签

# 检查数据是否有丢失情况
# print df.info()

# train_test_split
def train_test_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices,k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df,test_df

# print df.index
# RangeIndex(start=0, stop=150, step=1)
# print indices
# print res
# [123, 8, 90, 94, 120, 13, 110, 62, 104, 96, 21, 63, 1, 115, 16, 97, 122, 98, 54, 136]
# train_df,test_df = train_test_split(df,test_size=20)
# print test_df.head()

random.seed(0)
train_df,test_df = train_test_split(df,test_size=20)
### 工具类
data = train_df.values
# print data[:5]

# 数据纯度

def check_purity(data):
    label_colum = data[:,-1]
    unique_classes = np.unique(label_colum)
    # print unique_classes
    if len(unique_classes) == 1:
        return True
    else:
        return False

# 获取数据最后一列，也就是我们数据的标签
# print data[:,-1]

# print check_purity(train_df.values)
# print check_purity(train_df[train_df.petal_width < 0.8].values)


# 分类器
def classify_data(data):
    label_colum = data[:,-1]
    # 
    unique_classes,conts_unique_classes = np.unique(label_colum,return_counts=True)
    # 获取最大值
    index = conts_unique_classes.argmax()
    # 
    classification = unique_classes[index]
    return classification

#测试
# print classify_data(train_df[train_df.petal_width > 1.2].values)

# print unique_classes
# classification = label_colum[0]
# print classification

# Potential Splits?
def get_potential_splits(data):
    # 将返回字典类型数据结构
    # potential_splits = {3:[2.5,2.7]}
    potential_splits = {}
# 通过 data.shape (130,5) 获取数据列数量
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        # 对于每一个列创建空数组，用于放置分离数据阈值
        potential_splits[column_index] = []
        # 获取指定列的数据
        values = data[:,column_index]
        unique_values = np.unique(values)
        # 我们尝试打印
        # if column_index == 3:
        #     print unique_values
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index-1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].append(potential_split) 
    return potential_splits


# print data.shape 
# (130, 5)

potential_splits = get_potential_splits(train_df.values)
# sns.lmplot(data=train_df,x="petal_width",y="petal_length",hue="label",
#     fit_reg=False,size=6,aspect=1.5)
# plt.vlines(x=potential_splits[3],ymin=1,ymax=7)
# plt.hlines(y=potential_splits[2],xmin=0,xmax=2.5)
# plt.show()

# 分割数据
def split_data(data,split_column,split_value):
    split_column_values =  data[:,split_column]
    deta_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]
    return deta_below,data_above

split_column = 3
split_value = 0.8
# split_value = 1.05

data_below,data_above = split_data(data,split_column,split_value)
# data_below
# plotting_df = pd.DataFrame(data,columns=df.columns)
# sns.lmplot(data=plotting_df,x="petal_width",y="petal_length",fit_reg=False,size=6,aspect=1.5)
# plt.vlines(x=split_value,ymin=1,ymax=7)
# plt.show()

# 信息增益
def calculate_entropy(data):
    label_colum = data[:,-1]
    _, counts = np.unique(label_colum,return_counts=True)
    
    probabilities = counts/(counts.sum()*1.0)
    entropy = sum(probabilities * -np.log2(probabilities))

    # 返回信息熵
    return entropy

'''
计算熵

# 获取标签类
label_colum = data[:,-1]
# 返回值两个数组，需要第二参数为对每一个分类统计数量
_, counts = np.unique(label_colum,return_counts=True)
# [45 43 42]
# print counts
# print counts.sum()
# print counts/(counts.sum()*1.0)
# 获取每一个分类占总数的比例
probabilities = counts/(counts.sum()*1.0)
# $$ \sum_i^c P_i *(-log_2^{P_i})$$
entropy = probabilities * -np.log2(probabilities)

'''

'''
测试计算信息熵方法
print calculate_entropy(data_below)
# 0
print calculate_entropy(data_above)
# 0.8
# 0.0
# 0.9999001572094884
# 1.05
# 0.5225593745369408
# 0.9971085167216717
'''

# $$overall Entropy = \sum_{j=1}^2 p_j \cdot Entropy $$
def calculate_overall_entropy(data_below,data_above):
    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    overall_entropy = (p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above))
    return overall_entropy

'''
测试信息增益
print(calculate_overall_entropy(data_below,data_above))
# 1.05
# 0.8038152432539343
# 0.8
# 0.6461538461538462
'''

def determine_best_split(data,potential_splits):
    overall_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below,data_above = split_data(data,split_column=column_index,split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below,data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    return best_split_column,best_split_value

'''
测试
# (2, 2.5999999999999996)
print(determine_best_split(data,potential_splits))
'''
# sub_tree = {question:[yes_answer,no_answer]}

# Decision Tree Algorithm
# example_tree =  {"petal_width <- 0.8":["Iris-setosa",{"petal_width <= 1.65":[{"petal_length <= 4.9":["Iris-versicolor","Iris-virginical"]},"Iris-virginica"]}]}

# 算法
def decision_tree_algorithm(df,counter=0):
    # 转换类型 numpy array
    if counter == 0:
        data = df.values
    else:
        data = df

    # 
    if check_purity(data):
        classification = classify_data(data)
        return classification
    # 递归
    else:
        counter += 1
        # 
        potential_splits = get_potential_splits(data)
        split_column,split_value = determine_best_split(data,potential_splits)
        data_below,data_above = split_data(data,split_column,split_value)

        # 创建分支
        question = "{} <= {}".format(split_column,split_value)
        sub_tree = {question:[]}

        # find answer (recursion)
        yes_answer = decision_tree_algorithm(data_below,counter)
        no_answer = decision_tree_algorithm(data_above,counter)

        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)

        return sub_tree

'''
tree = decision_tree_algorithm(train_df[train_df.label != "Iris-virginica"])
print(tree)
{'3 <= 0.8': ['Iris-setosa', 'Iris-versicolor']}
'''