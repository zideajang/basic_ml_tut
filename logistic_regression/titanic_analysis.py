# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import sklearn.preprocessing as preprocessing

train_data = pd.read_csv('data/train.csv')
print train_data.head(3)
# print df.groupby('survived').count()

def set_missing_ages(p_df):
    p_df.loc[(p_df.age.isnull()),'age'] = p_df.age.dropna().mean()
    return p_df

# 处理缺失 age 的数据
df = set_missing_ages(train_data)
# print train_data.head(3)

# 处理归一化数值数据
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(train_data['age'].values.reshape(-1, 1))
df['age_scaled'] = scaler.fit_transform(train_data['age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(train_data['fare'].values.reshape(-1, 1))
df['fare_scaled'] = scaler.fit_transform(train_data['fare'].values.reshape(-1, 1), fare_scale_param)

# df['age_scaled'] = scaler.fit_transform(train_data['age'])
# df['fare_scaled'] = scaler.fit_transform(train_data['fare'])

# print(df.head(3))



# 处理类别意义的特征
def set_cabin_type(p_df):
    p_df.loc[(p_df.cabin.notnull()),'cabin'] = "Yes"
    p_df.loc[(p_df.cabin.isnull()),'cabin'] = "No"
    return p_df

df = set_cabin_type(df)

dummies_pclass = pd.get_dummies(train_data['pclass'],prefix='pclass')
# print dummies_pclass.head(3)
dummies_embarked = pd.get_dummies(train_data['embarked'],prefix='embarked')
# print dummies_embarked.loc[61]
dummies_sex = pd.get_dummies(train_data['sex'],prefix='sex')
# print dummies_sex.head(3)

df = pd.concat([df,dummies_embarked,dummies_sex,dummies_pclass],axis=1)
df.drop(['pclass','name','sex','ticket','cabin','embarked'],axis=1,inplace=True)

train_df = df.filter(regex='survived|age_.*|sibsp|parch|fare_.*|cabin_.*|embarked_.*|pclass_.*')
print train_df.head(1)
'''
   survived  sibsp  parch  age_scaled  fare_scaled  embarked_C  embarked_Q  embarked_S  pclass_1  pclass_2  pclass_3
0         0      1      0   -0.592481    -0.502445           0           0           1         0         0         1
'''


# from abupy import AbuML

# 对于 age 按小于 10 进行划分出新的 child 特征
# df['child'] = (train_data['age'] <= 10).astype(int)
# 年龄平方特征
# df['age*age'] = train_data['age'] * train_data['age']
# 归一化
# df['age*age_scaled'] = 
# print(df.head(3))
    