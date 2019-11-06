# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt

female_color = "#FA0000"

df = pd.read_csv('data/train.csv')
fig = plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,0))
df.survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3,4),(0,1))
df.survived[df.sex=="male"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Men Survived")

plt.subplot2grid((3,4),(0,2))
df.survived[df.sex=="female"].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=female_color)
plt.title("Women Survived")

plt.subplot2grid((3,4),(0,3))
df.sex[df.survived==1].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=[female_color,'b'])
plt.title("Sex of Survived")

plt.subplot2grid((3,4),(1,0),colspan=4)
for x in [1,2,3]:
    df.survived[df.pclass == x].plot(kind="kde")
plt.title("Class wrt Sruvived")
plt.legend(("1st","2nd","3rd"))


plt.subplot2grid((3,4),(2,0))
df.survived[(df.sex== "male") & (df.pclass ==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=[female_color,'b'])
plt.title("Rich Men of Survived")

plt.subplot2grid((3,4),(2,1))
df.survived[(df.sex== "male") & (df.pclass ==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=[female_color,'b'])
plt.title("Poor Men of Survived")

plt.subplot2grid((3,4),(2,2))
df.survived[(df.sex== "female") & (df.pclass ==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=[female_color,'b'])
plt.title("Rich Men of Survived")

plt.subplot2grid((3,4),(2,3))
df.survived[(df.sex== "female") & (df.pclass ==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=[female_color,'b'])
plt.title("Poor Men of Survived")

plt.show()