# coding=utf-8

import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import utils
from sklearn import linear_model,preprocessing,tree,model_selection
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from collections import namedtuple

def build_neural_network(hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training=tf.Variable(True,dtype=tf.bool)
    
    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc=tf.layers.batch_normalization(fc, training=is_training)
    fc=tf.nn.relu(fc)
    
    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        
        yield batch_x,batch_y

def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y

from sklearn.preprocessing import LabelEncoder
def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male","female"])
    data["sex"]=le.transform(data["sex"]) 
    return data

def nan_padding(data, columns):
    for column in columns:
        imputer=Imputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data

def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data

def tf_demo():
    train_data = pd.read_csv(r"./data/train.csv")
    test_data = pd.read_csv(r"./data/test.csv")

    nan_columns = ["age", "sibsp", "parch"]

    train_data = nan_padding(train_data, nan_columns)
    test_data = nan_padding(test_data, nan_columns)

    # test_passenger_id=test_data["passengerid"]
    not_concerned_columns = ["name", "ticket", "fare", "cabin", "embarked"]
    train_data = drop_not_concerned(train_data, not_concerned_columns)
    test_data = drop_not_concerned(test_data, not_concerned_columns)

    dummy_columns = ["pclass"]
    train_data=dummy_data(train_data, dummy_columns)
    test_data=dummy_data(test_data, dummy_columns)

    train_data = sex_to_int(train_data)
    test_data = sex_to_int(test_data)

    train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)

    model = build_neural_network()
    
    epochs = 200
    train_collect = 50
    train_print=train_collect*2

    learning_rate_value = 0.001
    batch_size=16

    x_collect = []
    train_loss_collect = []
    train_acc_collect = []
    valid_loss_collect = []
    valid_acc_collect = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iteration=0
        for e in range(epochs):
            for batch_x,batch_y in get_batch(train_x,train_y,batch_size):
                iteration+=1
                feed = {model.inputs: train_x,
                        model.labels: train_y,
                        model.learning_rate: learning_rate_value,
                        model.is_training:True
                    }

                train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
                
                if iteration % train_collect == 0:
                    x_collect.append(e)
                    train_loss_collect.append(train_loss)
                    train_acc_collect.append(train_acc)

                    if iteration % train_print==0:
                        print("Epoch: {}/{}".format(e + 1, epochs),
                        "Train Loss: {:.4f}".format(train_loss),
                        "Train Acc: {:.4f}".format(train_acc))
                            
                    feed = {model.inputs: valid_x,
                            model.labels: valid_y,
                            model.is_training:False
                        }
                    val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                    valid_loss_collect.append(val_loss)
                    valid_acc_collect.append(val_acc)
                    
                    if iteration % train_print==0:
                        print("Epoch: {}/{}".format(e + 1, epochs),
                        "Validation Loss: {:.4f}".format(val_loss),
                        "Validation Acc: {:.4f}".format(val_acc))
                    

        saver.save(sess, "./titanic.ckpt")

    # print("train_x:{}".format(train_x.shape))
    # print("train_y:{}".format(train_y.shape))
    # print("train_y content:{}".format(train_y[:3]))

    # print("valid_x:{}".format(valid_x.shape))
    # print("valid_y:{}".format(valid_y.shape))

    # print train_data.head()

def clean_data(data):
    data["fare"] = data["fare"].fillna(data["fare"].dropna().median())
    data["age"] = data["age"].fillna(data["age"].dropna().median())

    data.loc[data["sex"] == "male","sex"] = 0
    data.loc[data["sex"] == "female","sex"] = 1

    data["embarked"] = data["embarked"].fillna("S")
    data.loc[data["embarked"] == "S","embarked"] = 0
    data.loc[data["embarked"] == "C","embarked"] = 1
    data.loc[data["embarked"] == "Q","embarked"] = 2


def log_reg_model():
    train = pd.read_csv("data/train.csv")
    clean_data(train)
    
    # 定义标签
    target = train["survived"].values
    # 定义特征值
    # features = train[["pclass","sex","sibsp","parch"]].values
    features = train[["pclass","age","fare","embarked","sex","sibsp","parch"]].values

    classifier = linear_model.LogisticRegression()
    classifier_ = classifier.fit(features,target)

    print classifier_.score(features,target) #0.8002244668911336(first features) 0.7991021324354658(second features)

    poly = preprocessing.PolynomialFeatures(degree=2)
    poly_features = poly.fit_transform(features)

    classifier_ = classifier.fit(poly_features,target)
    print classifier_.score(poly_features,target) 
    
    '''
    0.7991021324354658
    0.835016835016835
    '''
def decision_tree_model():
    train = pd.read_csv("data/train.csv")
    clean_data(train)
    
    # 定义标签
    target = train["survived"].values
    # 定义特征值
    # features = train[["pclass","sex","sibsp","parch"]].values
    feature_names = ["pclass","age","fare","embarked","sex","sibsp","parch"]
    features = train[feature_names].values
    decision_tree = tree.DecisionTreeClassifier(random_state=1)
    decision_tree_ = decision_tree.fit(features,target)

    # 看到这个结果可能会联想到过拟合
    print decision_tree_.score(features,target) #0.9797979797979798
    scores = model_selection.cross_val_score(decision_tree,features,target,scoring='accuracy',cv=50)
    # print scores
    print scores.mean() #0.7848856209150326

    decision_tree = tree.DecisionTreeClassifier(random_state=1,max_depth=7,min_samples_split=2)
    decision_tree_ = decision_tree.fit(features,target)

def generalized_tree_model():
    train = pd.read_csv("data/train.csv")
    clean_data(train)
    
    # 定义标签
    target = train["survived"].values
    # 定义特征值
    # features = train[["pclass","sex","sibsp","parch"]].values
    feature_names = ["pclass","age","fare","embarked","sex","sibsp","parch"]
    features = train[feature_names].values

    # 看到这个结果可能会联想到过拟合
    # print scores

    generalized_tree = tree.DecisionTreeClassifier(random_state=1,max_depth=7,min_samples_split=2)
    generalized_tree_ = generalized_tree.fit(features,target)
    scores = model_selection.cross_val_score(generalized_tree,features,target,scoring='accuracy',cv=50)
    print scores.mean() #0.8243709150326798

    tree.export_graphviz(generalized_tree_,feature_names=feature_names,out_file="tree.dot")

    print decision_tree_.score(features,target) #0.9797979797979798

def run():
    # generalized_tree_model()
    tf_demo()

if __name__ == "__main__":
    run()