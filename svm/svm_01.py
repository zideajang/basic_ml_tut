# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run():
    iris_feature = u'花萼长度',u'花萼宽度',u'花瓣长度',u'花瓣宽度'
    path = 'data/iris.csv'
    data = pd.read_csv(path,header=None)
    x,y = data[range(4)],data[4]
    y = pd.Categorical(y).codes
    x = x[[0,1]]
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.6)
    # 分类器
    clf = svm.SVC(C=0.1,kernel='linear',decision_function_shape='ovr')

    clf.fit(x_train,y_train.ravel())

    # 准确度
    grid_hat = clf.predict(grid_test)
    

if __name__ == "__main__":
    run()