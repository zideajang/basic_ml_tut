#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 数据提供
data_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = data_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
# keras 

# 评估