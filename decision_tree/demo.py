# coding=utf-8
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline

df = pd.read_csv('./data/ad.data',header=None)

# print df.columns.values
explanatory_variable_columns = set(df.columns.values)
# print explanatory_variable_columns
explanatory_variable_columns.remove(len(df.columns.values) - 1)

response_variable_column = df[len(df.columns.values) - 1]