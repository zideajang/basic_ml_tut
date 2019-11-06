# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')
fig = plt.figure(figsize=(18,6))
plt.subplot2grid((2,3),(0,0))

df.survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.survived,df.age,alpha=0.1)
plt.title("Age wrt Survived")

plt.subplot2grid((2,3),(0,2))
df.pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("class")


# 根据这个
plt.subplot2grid((2,3),(1,0),colspan=2)
for x in [1,2,3]:
    df.age[df.pclass == x].plot(kind="kde")
plt.title("class wrt age")
plt.legend(("1st","2nd","3rd"))

plt.subplot2grid((2,3),(1,2))
df.embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Embarked")


plt.show()

