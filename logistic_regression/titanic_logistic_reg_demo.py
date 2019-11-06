# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

# 在开始之前我们先哀悼那些在这次海难中遇难的人们。

train = pd.read_csv('./data/train.csv')

# 查看数据基本信息
# print train.head()
# print train.count()
# print train.info()

# 查看数据丢失情况
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# 查看幸存者中男女比例
# sns.set_style('whitegrid')
# sns.countplot(x='survived', hue='sex', data=train, palette='RdBu_r')

# 查看幸存者中旅客等级
# sns.countplot(x='survived', hue='pclass', data=train)

# 查看乘客的年龄分布
# sns.distplot(train['age'].dropna(),kde=False,bins=30,color='Green')

# 查看乘客中兄弟姐妹配偶比例
# sns.countplot(x='sibsp',data=train)

# 
# plt.figure(figsize=(12,7))
# sns.boxplot(x='pclass',y='age',data=train, palette='winter')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['age'] = train[['age','pclass']].apply(impute_age,axis=1)

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# 删除掉 cabin 列
train.drop('cabin',axis=1,inplace=True)
train.dropna(inplace=True)
# print(train.head())

sex = pd.get_dummies(train['sex'],drop_first=True)
embark = pd.get_dummies(train['embarked'],drop_first=True)

# 删除 sex embarked name 和 tickets 列
train.drop(['sex','embarked','name','ticket'],axis=1,inplace=True)

# 将处理过的 sex 和 embark 列添加数据
train = pd.concat([train,sex,embark],axis=1)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train.drop('survived',axis=1),train['survived'],test_size=0.30,random_state=101)

# y 是要预测的真实值，其他作为特征值，把测试数据集大小设置比例 30% 设置随机状态

# 训练模型
# 导入 sklearn 的 LogisticRegression 进行处理
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
# 预测
predictions = logmodel.predict(X_test)
# 评估
print classification_report(y_test,predictions)

# 查看更新后数据
# print train.head()

# plt.show()