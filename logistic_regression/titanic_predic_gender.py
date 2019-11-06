# coding=utf-8

import pandas as pd

train = pd.read_csv("data/train.csv")

train["Hyp"] = 0
train.loc[train.sex == "female","Hyp"] = 1

train["Result"] = 0
train.loc[train.survived==train["Hyp"],"Result"] = 1
print train["Result"].value_counts()