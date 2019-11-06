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


逻辑回归的损失函数
在逻辑回归中代价函数

![logistic_reg_loss_fun.png](https://upload-images.jianshu.io/upload_images/8207483-827ace52a58f28ad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



我们结合这张图来看一下如何在逻辑回归模型定义评价的损失函数。

$$ -log(h_{\theta}(x)) $$
当 $y=1,h_{\theta}(x) = 1$ 时 $cost=0$
当 $y=1,h_{\theta}(x) = 0$ 时 $cost=\infty$

先看图中(蓝色)曲线表示当真实值为 1 时候损失函数多对应曲线，不难看粗当代价函数越接近 1 (也就是和真实值一致)损失函数就趋近于 0 ，相反代价函数趋近于无穷大，这就符合我们对于损失函数的要求。

$$ -\log(1-h_{\theta}(x))$$

当 $y=0,h_{\theta}(x) = 1$ 时 $cost=\infty$
当 $y=0,h_{\theta}(x) = 0$ 时 $cost=0$

先看图中(红色)曲线表示当真实值为 0 时候损失函数多对应曲线，不难看粗当代价函数越接近 0 (也就是和真实值一致)损失函数就趋近于 0 ，相反代价函数趋近于无穷大，这就符合我们对于损失函数的要求。

然后我们使用一个小技巧将两个分段函数合并为一个。通过一个方程我们将一个分段不可导的函数合并为一个可导的连续函数，接下来我们就可以用梯度下降优化方法迭代更新我们的参数。
$$ L(h_{\theta},y) = -y \log(h_{\theta}) - (1 - y) \log(1 - h_{\theta}(x))$$

当 y 取 0 或 1 情况下得到不同方程，这个和上面的函数表达形式一样。
$$ \begin{cases}
    L(h_{\theta},y) = - \log(h_{\theta}) & y= 1 \\
    L(h_{\theta},y) = - \log(1 - h_{\theta}(x)) & y = 0
\end{cases} $$

然后我们开始求解逻辑回归也是使用梯度下降法，要做梯度下降我们就需要让让损失函数对参数进行求导。
$$ L(h_{\theta},y) = - \frac{1}{m} \sum_{i=1}^N y \log(h_{\theta}) + (1 - y) \log(1 - h_{\theta}(x)) $$

这里求导过程要比之前的线性逻辑回归要复杂一些，涉及复合函数求导。涉及到链式求导方式来解决这中函数，我们在开始正式求导前，有必要复习一下求导中一些小技巧，这里特别注意一下，就是有关 sigmoid 函数求导函数为了最后一个方程，这里大家只要知道 sigmoid 函数的导数函数的模样，具体如何推导来的大家可以自己查查下资料。

$$ \frac{\partial L(\theta)}{\partial h_{\theta}(x)} \frac{\partial h_{\theta}(x)}{\partial \theta}$$

首先我们对 log 进行求导,$\log f(x) $ 的导数就是 
$$\frac{1}{f(x)}$$

$$
\begin{cases}
    h_{\theta}(x) = g(\theta^Tx) \\
    g(x) = \frac{1}{1 + e^{-x}} \\
    h_{\theta} \prime = h_{\theta}(x)(1 - h_{\theta}(x))
\end{cases} $$
首先我们看一下上面这些方程，只有理解这些方程含义和由来我们才能真正理解接下来逻辑回归求导过程。

$$ \frac{\partial(h_{\theta}(x),y)}{\theta \theta} = - \frac{{1}}{m} \sum_{i=1}^m(\frac{y}{h_{\theta}(x)} - \frac{(1-y)}{1 - h_{\theta}(x)} ) \frac{\partial h_{\theta}(x)}{\partial \theta}$$

对方程进行化简，这里很简单我就不啰嗦了。

$$ \frac{\partial(h_{\theta}(x),y)}{\theta \theta} = - \frac{{1}}{m} \sum_{i=1}^m(\frac{y(1 - h_{\theta}(x)) - h_{\theta}(x)(1-y)  }{h_{\theta}(x)(1 - h_{\theta}(x))} ) \frac{\partial h_{\theta}(x)}{\partial \theta}$$

化简之后将$h_{\theta} \prime = h_{\theta}(x)(1 - h_{\theta}(x)$ 将这个已知 sigmoid 带入方程进行化简。

$$ \frac{\partial(h_{\theta}(x),y)}{\theta \theta} = - \frac{{1}}{m} \sum_{i=1}^m(\frac{y - h_{\theta}(x) }{h_{\theta}(x)(1 - h_{\theta}(x))} ) \frac{\partial h_{\theta}(x)}{\partial \theta}$$

$$ \frac{\partial(h_{\theta}(x),y)}{\theta \theta} = - \frac{{1}}{m} \sum_{i=1}^m(\frac{y - h_{\theta}(x) }{h_{\theta}(x)(1 - h_{\theta}(x))} )  h_{\theta}(x)(1 - h_{\theta}(x))x$$

经过一些列化简后我们就得到想要得到梯度下降的优化参数方程。

$$ \frac{\partial(h_{\theta}(x),y)}{\theta \theta} =   \frac{{1}}{m} \sum_{i=1}^m x(h_{\theta}(x) - y)$$

有了上面求导过程我们就不难理解上一次分享实例，大家可以自己看一下上一次最后那个例子，尝试理解一下，其实机器学习中代码和实现比较好理解，难理解的是其背后原理。而且在机器学习中了解其算法和模型设计背后思想是十分必要，只有理解其原理才能通过调参数来训练出来好的模型​。​

有了优化，我们现在需要对结果进行评估。
### 正确率和召回率
正确率(Precsion)和召回率(Recall)是广泛应用于信息检索和统计学分类领域的两个度量值，用来评估结果的质量。

一般来说，正确率就是检索出来的条目有多少是正确的，召回率就是所有正确的条目有多少被检索出来
$$ F1值 = 2 \times \frac{正确率 \times 召回率}{正确率 + 召回率} $$
是综合上面两个指标的估计指标，用于反映整体的指标

这几个指标的取值都在 0 - 1 之间，数值越接近于 1 效果越好。

有三类事物分别为 A，B 和 C 他们数量分别是 1400，300，300 我们目标获得更多的 A，进行一次收集后得到 A B 和 C 数量分别是 700，200 和 100 那么

$$ 正确率 = \frac{700}{700 + 200 + 100} = 70 \% $$
$$ 召回率 = \frac{700}{1400} = 50 \% $$
$$ F1值 = \frac{0.7 \times 0.5 \times 2}{0.7 + 0.5} = 58.3 \% $$





$$ 正确率 = \frac{1400}{1400 + 300 + 300} = 70 \% $$
$$ 召回率 = \frac{1400}{1400} = 100 \% $$
$$ F1值 = \frac{0.7 \times 1 \times 2}{0.7 + 1} = 82.35 \% $$

我们希望检索的结果的正确率越高越好，同时也希望召回率越高越好不过在某些情况他们之间是矛盾的。有的时候我们想要提高准确率就会影响召回率，反之亦然。

正式因为正确率和召回率指标有时候出现的矛盾原因，我们可以通过 F-Measue 来综合考虑指标这时候我们就可以用 F1值来综合考虑。

其实上面介绍 F1公式是这个 $\beta = 1 $公式的特例。

$$ F_{\beta} = (1 + \beta^2) \dot \frac{准确率 \times 召回率}{(\beta^2 \times 准确率 ) + 召回率} $$


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
读取训练数据集，我们打印每一个列数据量，可以查看出某些列数据并不完整，carbin 只有在 204 条记录中有值。
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

![titanic_missing_data.png](https://upload-images.jianshu.io/upload_images/8207483-d6ed3fc216ffafed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

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

)

![titanic_missing_data.png](https://upload-images.jianshu.io/upload_images/8207483-d6ed3fc216ffafed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从热力图上来看，大约 20% 年龄数据丢失，在机舱(carbin)列丢失数据过多在清洗数据会将其删除。


![titanic_survived_bar.png](https://upload-images.jianshu.io/upload_images/8207483-6e91c3fb9115b156.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过上面直方图可以看出遇难者(用 0 表示遇难)为 60% 而幸存者(用 1 表示)占到 40%

我们

```python
df.survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.survived,df.age,alpha=0.1)
plt.title("Age wrt Survived")
```

![幸存者和遇难者年龄分布图](https://upload-images.jianshu.io/upload_images/8207483-7ff2cbcb924d9bb5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在图Age wrt Survived 中横坐标 0 代表遇难者 1 代表幸存者，可以发现幸存者年龄要小于遇难者的年龄。

```python
plt.subplot2grid((2,3),(0,2))
df.pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("class")
```

![客舱等级](https://upload-images.jianshu.io/upload_images/8207483-8501a318f5ba8bb3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![图](https://upload-images.jianshu.io/upload_images/8207483-a63e7485de93ae1e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

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