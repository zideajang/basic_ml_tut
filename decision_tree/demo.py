# coding=utf-8
import pandas as pd

# 引入决策树分类器
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from sklearn import datasets

def demo_one():
    df = pd.read_csv('./data/ad.data',header=None)

    # print df.columns.values
    explanatory_variable_columns = set(df.columns.values)
    # print explanatory_variable_columns
    explanatory_variable_columns.remove(len(df.columns.values) - 1)

    response_variable_column = df[len(df.columns.values) - 1]

def demo_two():
    iris = datasets.load_iris()
    # 保留后两个特征
    X = iris.data[:,2:]
    y = iris.target

    plt.scatter(X[y==0,0],X[y==0,1])
    plt.scatter(X[y==1,0],X[y==1,1])
    plt.scatter(X[y==2,0],X[y==2,1])
    plt.show()
      
    dt_clf = DecisionTreeClassifier(max_depth=2,criterion="entropy")
    dt_clf.fit(X,y)
    
def plot_decision_boundary():
    

def run():
    demo_two()

if __name__ == "__main__":
    run()