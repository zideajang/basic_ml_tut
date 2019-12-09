# coding=utf-8

import pandas as pd
# import utils
from sklearn import linear_model,preprocessing,tree,model_selection

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
    generalized_tree_model()

if __name__ == "__main__":
    run()