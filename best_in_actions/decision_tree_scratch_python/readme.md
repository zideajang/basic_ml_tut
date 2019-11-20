
### 引入依赖
```python
# 导入依赖
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint
#  加载并准备数据
names = ["sepal_length","sepal_width","petal_length","petal_width","species"]
df = pd.read_csv("data/iris.csv",header=None,names=names)
print df.head()
```


```
   sepal_length  sepal_width  petal_length  petal_width      species
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
```
去掉 id 列

```python
def train_test_split(df,test_size):

    return train_df,test_df
```

```python
test_size = 20
if isinstance(test_size,float):
    test_size = round(test_size * len(df))
indices = df.index.tolist()
test_indices = random.sample(population=indices,k=test_size)
test_df = df.loc[test_indices]
train_df = df.drop(test_indices)
# print indices
# print res
# [123, 8, 90, 94, 120, 13, 110, 62, 104, 96, 21, 63, 1, 115, 16, 97, 122, 98, 54, 136]
```

### 定义划分数据集方法
```python
def train_test_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices,k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df,test_df
```
测试方法
```python
train_df,test_df = train_test_split(df,test_size=20)
print test_df.head()
```

```
     sepal_length  sepal_width  petal_length  petal_width            label
29            4.7          3.2           1.6          0.2      Iris-setosa
130           7.4          2.8           6.1          1.9   Iris-virginica
56            6.3          3.3           4.7          1.6  Iris-versicolor
98            5.1          2.5           3.0          1.1  Iris-versicolor
135           7.7          3.0           6.1          2.3   Iris-virginica
```

```python
random.seed(0)
```

### 第二节

```python
def check_purity(data):

    return False
```

```python
label_colum = data[:,-1]
unique_classes = np.unique(label_colum)
print unique_classes
```

```
['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
```

```python
if len(unique_classes) == 1:
    return True
else:
    return False
```

#### 检查纯度方法
```python
def check_purity(data):
    label_colum = data[:,-1]
    unique_classes = np.unique(label_colum)
    # print unique_classes
    if len(unique_classes) == 1:
        return True
    else:
        return False
```

```
print check_purity(train_df.values)
```
```
False
```

```python
print check_purity(train_df[train_df.petal_width < 0.8].values)
```
```
True
```

### 分类器
```
def classify_data(data):
    return classification
```

```
label_colum = data[:,-1]
print np.unique(label_colum,return_counts=True)
```
```
(array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object), array([45, 43, 42]))
```

```python
def classify_data(data):
    label_colum = data[:,-1]
    unique_classes,conts_unique_classes = np.unique(label_colum,return_counts=True)
    index = conts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification
```

```python
print classify_data(train_df[train_df.petal_width < 0.8].values)

```
```
Iris-setosa
```

```python
print classify_data(train_df[train_df.petal_width > 1.2].values)

```
```
Iris-virginica
```

```python
def get_potential_splits(data):
    return potential_splits
```

```python
potential_splits = {}
_, n_columns = data.shape
for column_index in range(n_columns - 1):
    potential_splits[column_index] = []
    values = data[:,column_index]
    if column_index == 3:
        print values
```

```python
# potential_splits = {3:[2.5,2.7]}
potential_splits = {}
# 通过 data.shape (130,5) 获取数据列数量
_, n_columns = data.shape
for column_index in range(n_columns - 1):
    # 对于每一个列创建空数组，用于放置分离数据阈值
    potential_splits[column_index] = []
    # 获取指定列的数据
    values = data[:,column_index]
    # 我们尝试打印
    if column_index == 3:
        print values
```


```
[0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 0.2 0.2 0.1 0.1 0.2 0.4 0.4 0.3
 0.3 0.3 0.2 0.4 0.2 0.5 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 0.1 0.2 0.1 0.2
 0.2 0.2 0.3 0.3 0.6 0.4 0.2 0.2 0.2 1.4 1.5 1.5 1.3 1.5 1.3 1.6 1.0 1.3
 1.4 1.5 1.0 1.3 1.4 1.5 1.0 1.5 1.1 1.8 1.5 1.2 1.3 1.7 1.5 1.0 1.1 1.0
 1.2 1.6 1.5 1.6 1.5 1.3 1.3 1.2 1.4 1.0 1.3 1.2 1.3 1.3 1.1 1.3 2.5 1.9
 2.1 1.8 2.2 2.1 1.7 1.8 1.8 2.0 1.9 2.1 2.4 2.3 1.8 2.3 1.5 2.3 2.0 1.8
 2.1 1.8 1.8 2.1 1.6 1.9 2.0 2.2 1.5 1.4 1.8 1.8 2.1 2.4 2.3 1.9 2.3 2.5
 2.3 1.9 2.3 1.8]
 ```

 ```python
 _, n_columns = data.shape
for column_index in range(n_columns - 1):
    # 对于每一个列创建空数组，用于放置分离数据阈值
    potential_splits[column_index] = []
    # 获取指定列的数据
    values = data[:,column_index]
    unique_values = np.unique(values)
    # 我们尝试打印
    if column_index == 3:
        print unique_values
```

 ```
 [0.1 0.2 0.3 0.4 0.5 0.6 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1
 2.2 2.3 2.4 2.5]
 ```

 ```python
 def get_potential_splits(data):
    # 将返回字典类型数据结构
    # potential_splits = {3:[2.5,2.7]}
    potential_splits = {}
# 通过 data.shape (130,5) 获取数据列数量
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        # 对于每一个列创建空数组，用于放置分离数据阈值
        potential_splits[column_index] = []
        # 获取指定列的数据
        values = data[:,column_index]
        unique_values = np.unique(values)
        # 我们尝试打印
        # if column_index == 3:
        #     print unique_values
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index-1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].append(potential_split) 
    return potential_splits
    ```

    ```python
    print get_potential_splits(train_df.values)
    ```

    ```
    {0: [4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95, 5.05, 5.15, 5.25, 5.35, 5.45, 5.55, 5.65, 5.75, 5.85, 5.95, 6.05, 6.15, 6.25, 6.35, 6.45, 6.55, 6.65, 6.75, 6.85, 6.95, 7.05, 7.15, 7.25, 7.35, 7.5, 7.65, 7.800000000000001], 1: [2.25, 2.3499999999999996, 2.45, 2.55, 2.6500000000000004, 2.75, 2.8499999999999996, 2.95, 3.05, 3.1500000000000004, 3.25, 3.3499999999999996, 3.45, 3.55, 3.6500000000000004, 3.75, 3.8499999999999996, 3.95, 4.05, 4.15, 4.300000000000001], 2: [1.05, 1.15, 1.25, 1.35, 1.45,1.55, 1.65, 1
    ```

```python
potential_splits = get_potential_splits(train_df.values)
sns.lmplot(data=train_df,x="petal_width",y="petal_length")
plt.show()
```
    ![best_in_action_dt_01.png](https://upload-images.jianshu.io/upload_images/8207483-fc3540cbf6031734.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
sns.lmplot(data=train_df,x="petal_width",y="petal_length",hue="label",fit_reg=False)
plt.show()
```

![best_in_action_dt_02.png](https://upload-images.jianshu.io/upload_images/8207483-edd7131868eac576.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
potential_splits = get_potential_splits(train_df.values)
sns.lmplot(data=train_df,x="petal_width",y="petal_length",hue="label",
    fit_reg=False,size=6,aspect=1.5)
plt.show()
```
![best_in_action_dt_03.png](https://upload-images.jianshu.io/upload_images/8207483-65113bc1c94f5d3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
potential_splits = get_potential_splits(train_df.values)
sns.lmplot(data=train_df,x="petal_width",y="petal_length",hue="label",
    fit_reg=False,size=6,aspect=1.5)
plt.vlines(x=potential_splits[3],ymin=1,ymax=7)
plt.show()
```
![best_in_action_dt_05.png](https://upload-images.jianshu.io/upload_images/8207483-81d8080b9a793289.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
potential_splits = get_potential_splits(train_df.values)
sns.lmplot(data=train_df,x="petal_width",y="petal_length",hue="label",
    fit_reg=False,size=6,aspect=1.5)
plt.vlines(x=potential_splits[3],ymin=1,ymax=7)
plt.hlines(y=potential_splits[2],xmin=0,xmax=2.5)
plt.show()

```
![best_in_action_dt_06.png](https://upload-images.jianshu.io/upload_images/8207483-37e6b4089e460061.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 分割数据
```python
def split_data(data,split_column,split_value):
    return data_below,data_above
```

```python
split_column = 3
split_value = 0.8
split_column_values =  data[:,split_column]
```
```
[0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 0.2 0.2 0.1 0.1 0.2 0.4 0.4 0.3
 0.3 0.3 0.2 0.4 0.2 0.5 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 0.1 0.2 0.1 0.2
 0.2 0.2 0.3 0.3 0.6 0.4 0.2 0.2 0.2 1.4 1.5 1.5 1.3 1.5 1.3 1.6 1.0 1.3
 ```

 ```python
 split_column = 3
split_value = 0.8
split_column_values =  data[:,split_column]
split_column_values <= split_value
print split_column_values <= split_value
```
```
[ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True False False False
 False False False False False False False False False False False False
```

```python
def split_data(data,split_column,split_value):
    split_column_values =  data[:,split_column]
    deta_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]
    return deta_below,data_above

split_column = 3
split_value = 0.8

data_below,data_above = split_data(data,split_column,split_value)
plotting_df = pd.DataFrame(data,columns=df.columns)
sns.lmplot(data=plotting_df,x="petal_width",y="petal_length",fit_reg=False,size=6,aspect=1.5)
plt.vlines(x=split_value,ymin=1,ymax=7)
plt.show()
```

![best_in_action_dt_08.png](https://upload-images.jianshu.io/upload_images/8207483-b2e9e039ddc59afd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 信息增益
```python
def calculate_entropy(data):
    return entropy
```

```python
label_colum = data[:,-1]
_, counts = np.unique(label_colum,return_counts=True)
# [45 43 42]
# print counts
# print counts.sum()
print counts/(counts.sum()*1.0)

```
