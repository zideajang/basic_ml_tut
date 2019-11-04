回归

垃圾邮件分类，判断用户办理信用卡等二分类问题。

线性回归(linear regression)是用来做回归(预测)的，逻辑回归(logistic regression)是用来做分类的。虽然两者用于解决不同问题，但是因为名字里都有回归所以还是有着一定关系不然就叫逻辑分类。


这个函数就是 sigmoid 函数，函数图如下图我们看一看他长什么样具体有哪些特征，取值在 0 - 1 之间以 0.5 为分界线小于零，当 x 趋近正无穷时候函数值 f(x) 就趋近于 1 ，相反当 x 趋近于负无穷时候 f(x) 就趋近于 0。
![sigmoid.jpeg](https://upload-images.jianshu.io/upload_images/8207483-504cff6ba17a892d.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

主要用于处理回归问题，逻辑回归用于分类问题。经典就是做为二分类问题。


决策边界


$$h_{\theta} = g(\theta_0 + \theta_1x_1 + \theta_2x_2)$$
$$ -3 + x_1 + x_3 \ge 0 $$

$$h_{\theta} = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3^2 + \theta_4x_2^2)$$

这样我们就是知道怎么把线性转换为分类问题，我们通过画一条直线让两边特征值带入方程根据函数值大于 0  或小于 0 通过 sigmoid 可以把大于 0 和小于 0 的函数值分布0 到 1 之间的概率分布。小于 0 经过 sigmoid 函数就会变成小于 0.5 我们这样就可以将其归为一类
二分类问题可以扩展到多分类问题


要做分类问题，从简单的二分类开始

标签(1/0)
可以使用最简单的单位阶跃


损失函数

预测值接近于 0 时候就是代价函数
当 $y=1,h_{\theta}(x) = 1$ 时 $cost=0$
当 $y=1,h_{\theta}(x) = 0$ 时 $cost=\infty$

当 $y=1,h_{\theta}(x) = 1$ 时 $cost=\infty$
当 $y=1,h_{\theta}(x) = 0$ 时 $cost=0$

有关线
### 泰坦尼克生与死
在开始之前我们先哀悼那些在这次海难中遇难的人们。
### 引入依赖
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
这里我们使用 pandas 的一个 python 数据库，pandas 好处就是可以将数据进行类似表格格式格式化。
### 准备数据
```python
train = pd.read_csv('./data/train.csv')
print train.head()
```

```
print train.count()
```
```
survived    891
pclass      891
name        891
sex         891
age         714
sibsp       891
parch       891
ticket      891
fare        891
cabin       204
embarked    889
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 11 columns):
survived    891 non-null int64
pclass      891 non-null int64
name        891 non-null object
sex         891 non-null object
age         714 non-null float64
sibsp       891 non-null int64
parch       891 non-null int64
ticket      891 non-null object
fare        891 non-null float64
cabin       204 non-null object
embarked    889 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 76.6+ KB
```
总共有可以看到 891 条记录，中间某些列中可能有一些空值，稍后将对其进行处理。

### 检查数据的丢失
使用 seaborn 来创建一个热力图，来检查丢失数据位置
```
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
```

![titanic_missing_data.png](https://upload-images.jianshu.io/upload_images/8207483-d6ed3fc216ffafed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从热力图上来看，大约 20% 年龄数据丢失，在机舱(carbin)列丢失数据过多在清洗数据会将其删除。

### 可视化来观察数据
#### 查看幸存者中男女比例

```python
sns.set_style('whitegrid')
sns.countplot(x='survived', hue='sex', data=train, palette='RdBu_r')
```

![survivied_sex](https://upload-images.jianshu.io/upload_images/8207483-56ade72b3621f3c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 查看幸存者乘客等级
![幸存者旅客等级](https://upload-images.jianshu.io/upload_images/8207483-286d81b59f9066b6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从图中我们可以看出在遇难者中处于等级 3 (也就是最低等级) 游客比例比较大。

#### 查看乘客年龄分布

![乘客年龄分布](https://upload-images.jianshu.io/upload_images/8207483-d428350c4a145944.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

年龄集中在 20 到 30 之间，

#### 查看乘客中兄弟姐妹配偶
![兄弟姐妹配偶](https://upload-images.jianshu.io/upload_images/8207483-b4f8372e9b184c21.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 查看票价分布

### 清洗数据
对于丢失年龄字段数据，选择填写年龄而不是删除这些数据，这里不是用乘客的平均年龄来填写而是通过乘客等级对平均年龄进行分类，然后根据乘客等级进行填写补充丢失年龄。
![titanic_01.png](https://upload-images.jianshu.io/upload_images/8207483-fd1006833770d8c0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从图中可以明显看出高等级乘客年龄往往较大，所有根据乘客等级来估计其年龄。

```python
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['age'] = train[['age','pclass']].apply(impute_age,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```

![titanic_clear_data.png](https://upload-images.jianshu.io/upload_images/8207483-f520146072fecd4b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
train.drop('cabin',axis=1,inplace=True)
train.dropna(inplace=True)
print(train.head())
```
### 转换分类特征

```
   survived  pclass   age  sibsp  parch     fare  male  Q  S
0         0       3  22.0      1      0   7.2500     1  0  1
1         1       1  38.0      1      0  71.2833     0  0  0
2         1       3  26.0      0      0   7.9250     0  0  1
3         1       1  35.0      1      0  53.1000     0  0  1
4         0       3  35.0      0      0   8.0500     1  0  1

```
#### 训练集合测试集
```python
from sklearn.model_selection import train_test_split
```

### 构建逻辑回归模型

### 训练和预测

```
              precision    recall  f1-score   support

           0       0.80      0.91      0.85       163
           1       0.82      0.65      0.73       104

   micro avg       0.81      0.81      0.81       267
   macro avg       0.81      0.78      0.79       267
weighted avg       0.81      0.81      0.80       267
```
通过结果可以看到得到了 81% 精确度

### 曲线回归
- 
$$ y = ln(ax + b) $$
$$ y \prime = e^y $$
$$ e^y = ax + b = y \prime $$

$$ y = \frac{a}{x} + b $$
$$ y = \frac{1}{1 + e(-ax + b))} $$

$$ y = ax + bx^2 + c $$
$$ x_1 = x $$
$$ x_2 = x^2 $$
$$ y =  ax_1 + bx_2 + c $$
面对这些曲线我们就无法用线性模型

$$ w^* = \arg \min $$

### 正则化(Regulariztion)
为什么需要线性回归，我们先复习一下线性回归最小二乘法
$$ J(\theta) = \frac{1}{2m}[\sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2] $$
$$ J(\theta) = \frac{1}{2m}[\sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n |\theta_j|] $$