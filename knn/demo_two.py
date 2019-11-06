# coding=utf-8
from math import *
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
# from sklearn import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def train(X_train, y_train):
    # do nothing
    return
def predict(X_train, y_train, x_test, k):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute and store L2 distance
        distances.append([np.sqrt(np.sum(np.square(x_test - X_train[i, :]))), i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return most common target
    return Counter(targets).most_common(1)[0][0]
def k_nearest_neighbor(X_train, y_train, X_test, k):
    # train on the input data
    train(X_train, y_train)

    # loop over all observations
    predictions = []
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))

    return np.asarray(predictions)
def knn_classifier():
    pass
def k_nearest_neighbor(X_train, y_train, X_test, predictions, k):
    # check if k larger than n
    assert k <= len(X_train), "[!] k can't be larger than number of samples."

    # train on the input data
    train(X_train, y_train)

	# predict for each testing observation
	for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))

def iris_classifier():
    print("iris classifier")
    names =  [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'class',
    ]
    df = pd.read_csv("data/iris.csv",header=None,names=names)
    # print(data.head())
    # print(data)
    X = np.array(df.ix[:, 0:4])
    y = np.array(df['class'])
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # 实例化学习模型(k = 3)
    knn = KNeighborsClassifier(n_neighbors=3)

    # 训练模型
    knn.fit(X_train, y_train)

    # 预测
    pred = knn.predict(X_test)

    # 评估准确度
    print("accuracy: {}".format(accuracy_score(y_test, pred)))

def car_classifier():
    data = pd.read_csv("data/car.csv")
    print(data.head())

    le = preprocessing.LabelEncoder()
    buying = le.fit_transform(list(data["buying"]))
    maint = le.fit_transform(list(data["maint"]))
    door = le.fit_transform(list(data["door"]))
    persons = le.fit_transform(list(data["persons"]))
    lug_boot = le.fit_transform(list(data["lug_boot"]))
    safety = le.fit_transform(list(data["safety"]))
    cls = le.fit_transform(list(data["class"]))
    
    predict = "class"  #optional

    X = list(zip(buying, maint, door, persons, lug_boot, safety))
    y = list(cls)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    model = KNeighborsClassifier(n_neighbors=9)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)

    predicted = model.predict(x_test)
    names = ["unacc", "acc", "good", "vgood"]

    for x in range(len(predicted)):
        print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
        n = model.kneighbors([x_test[x]], 9, True)
        print("N: ", n)
def show_table_header():
    data = pd.read_csv("data/car.csv")
    print(data.head())

# $$ \sqrt{\sum_{i=n}^N (x_i - y_i)^2}$$
def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))


def run():
    print(euclidean_distance([0,3,4,5],[7,6,3,-1]))

if __name__ == "__main__":
    # show_table_header()
    iris_classifier()