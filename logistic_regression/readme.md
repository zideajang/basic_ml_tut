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



### 泰坦尼克生与死
在开始内容之前，让我们先哀悼一下那些在这次海难中遇难的人们。1912 年 4 月 15 日，在泰坦尼克号首航，与冰山相撞后沉没，共 2224 名乘客和船员中 1502 人在这次沉船事件丧生。
通过机器学习分析什么样人生存概率。

#### 应用
现在可以大家已经掌握一些机器学习算法，现在可以通过做一个机器学习示例，综合之前学习过知识来解决一些实际问题。
- 数值归一化
- 回归预测

#### 引入依赖
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
这里我们使用 pandas ，pandas 是一个可以表格形式操作数据和展示数据的 python 常用库，pandas 好处就是可以将数据进行类似表格格式格式化。numpy 是对矩阵和向量操作比较方便的 python 库，其对矩阵数据结构的操作在速度上要优于 python 自带的方法和工具集。 

### 准备数据
```python
train = pd.read_csv('./data/train.csv')
print train.head()
```
| 特征名  | 说明  |
|---|---|
| survived  | 幸存者 1 表示生还 0 表示遇难   |
| pclass  | 客舱等级 1,2,3   |
| name  | 乘客姓名   |
| sex  | 性别   |
| age  | 年龄   |
| sibsp  | 兄弟姐妹/配偶数量   |
| parch  | 父母/孩子的数量   |
| ticket  | 船票号码   |
| fare  | 票价   |
| cabin  | 客舱   |
| embarked  | 登船的港口 C 表示瑟堡 Q皇后镇 S南安普顿   |


读取训练数据集，样本数量为 891 ，我们打印出每一个列的数据量，可以查看出某些列数据并不完整，carbin 只有在 204 条记录中有值。
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
当然我们还可以提供 info 方法来检查数据更详细的信息。

```python
data_train.info()
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

从图中，不难看出在遇难者中处于等级 3 (也就是最低等级) 游客比例比较大。因为这部分乘客年龄偏于年轻。

#### 查看乘客年龄分布

![乘客年龄分布](https://upload-images.jianshu.io/upload_images/8207483-d428350c4a145944.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

年龄集中在 20 到 30 之间，说明

#### 查看乘客中兄弟姐妹配偶
![兄弟姐妹配偶](https://upload-images.jianshu.io/upload_images/8207483-b4f8372e9b184c21.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 检查数据的丢失
使用 seaborn 来创建一个热力图，来检查丢失数据位置，这样通过图表查看感觉更加直观。
```
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
```

)

![titanic_missing_data.png](https://upload-images.jianshu.io/upload_images/8207483-d6ed3fc216ffafed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从热力图上来看，大约 20% 年龄数据丢失，在机舱(carbin)列丢失数据过多在清洗数据会将其删除。


![titanic_survived_bar.png](https://upload-images.jianshu.io/upload_images/8207483-6e91c3fb9115b156.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过上面直方图可以看出遇难者(用 0 表示遇难)为 60% 而幸存者(用 1 表示)占到 40%

#### 观察数据的特征
我们查看一下生存预测的数据标签，特征 passengerId, Ticket 和 Name 维度下的数据基本完全随机，所以不再特征值考虑范围内。剩下的特征可以分为数值意义的特征和类别意义的特征
- 数值 age sibsp parch
- 类别: pclass sex embarked cabin
#### 处理缺失的数据
这里因为我们数据样本比较珍贵，只有 891 ，所以不能使舍去丢失 age 数据的样本，而是需要我们填充数据来尽量还原，如何还原这些样本 age 数据，其实这也是一个预测问题，我们暂时用均值来代替随后，我们通过回归预测模型来预测年龄数据。

```python
def set_missing_ages(p_df):
    p_df.loc[(p_df.age.isnull(),'age')] = p_df.age.dropna().mean()
    return p_df
```

#### 归一化数值数据
对于两类特征数据混乱问题，有的时候我们不希望任意个特征对模型影响较大，因为特征所取值较大，那么在计算损失函数时候就会影响比较大，所以我们需要归一化来处理这样问题，让每一个特征对模型影响都是相同，这一点我想大家都能理解。数据归一化。

#### 处理类别意义的特征
在数据样本中 cabin 数据缺失还是比较严重，所以我们可以将这个问题简化为有没有的问题，也就是同 Yes 和 No 来取代之前数值
```python
def set_cabin_type(p_df):
    p_df.loc[(p_df.cabin.notnull()),'cabin'] = "Yes"
    p_df.loc[(p_df.cabin.isnull()),'cabin'] = "No"
    return p_df
```
在决策树学习和逻辑回归时候已经学习到用热编码(one-hot)来表示类别，其实类别表示问题上数值并不是恰当能够反映类别，我们需要通过热编码来表示类别。

```
   pclass_1  pclass_2  pclass_3
0         0         0         1
1         1         0         0
2         0         0         1
```

### 构造非线性特征
所谓线性模型就是把特征对分类结果的作用加起来，也就是线性模型能表示类似于 $y= x_1 + x_2$ 关系的表达式，其中 y 表示分类结果，$x_1,x_2$ 表示特征对分类作用。所以线性模型无法表达非线性关系，不过我们可以通过添加一些新的具有非线性的特征来弥补线性模型表现力不足的问题。
特征非线性分类两类
- 用于表达**数值特征**本身的非线性因素
- 用于表达特征与特征之间存在非线性关联，

也就是利用现有特征来创建一些更有表达力的特征

#### 评估特征作用
#### 构造特征的数学意义
上面我们通过很多技巧人工地构造一些非线性特征，可以弥补非线性模型表达不足。那么这些非线性特征是如何弥补线性模型不足的呢?我们现在就来回答这个问题。先给出答案然后再做解释，答案就是**低纬的非线性关系可以在高维空间上展开**。有时候我们数据一个低纬空间是不可分，当我们为通过增加特性来提高维度，他可能就是可分的。但是并不是代表特征越多越好，特征多了就容易出现过拟合，我们新建维度可以清楚将每一个训练集中数据样特征记住区分，那么对于新增样本（测试数据集）模型就变得束手无策了。
这是因为我们数据是有噪音，这些噪音对模型预测是没有帮助的，记住这噪音势必会影响模型泛化能力。


### 选择模型
在设计模型需要考虑两个问题就是**过拟合**和**欠拟合**的问题，他们都是由于模型复杂度不适当所带来问题。
#### 过拟合
我们可以在损失函数正价正则化来解决过拟合问题，也可以通过增加数据量或简化模型来解决过拟合问题。其实正则化问题也就是SVM中拉格朗日函数函数，作为对损失函数一种约束，确保损失函数与
#### 欠拟合
也就是我们模型设计过于简单，也就是模型能力不足。大家应该注意所谓**过拟合**和**欠拟合**,都是需要我们重新
### 模型调试


### 分类模型评估指标
评估模型，也就是合理定义评估指标，只有好的评估才能对模型的选择、调试有指导意义。这句话延伸就不做过多解释


## 回归模型

## 决策树模型
## 模型融合
我们都是知道“三个臭皮匠，顶一个诸葛亮”，有时候我们模型(例如决策树分类器)能力比较弱，我们需要将多个弱分类器组合起来完成一个任务。我们可以让多个个性差异的模型形成模型群体，共同发挥作用，从而获取更好的表现成绩，这就是模型融合(Ensamble)。好处是消除个性化，降低因个性差异带来错误。
在决策树我们通过极度丰富的特征和极深的树来构造出一颗在训练集上接近满分的决策树，但是容易发现这是过拟合的模型。在随机森林中我们模型是通过随机数据样本和随机选择特征来达到创建不同(个性)模型的目的的。
### 随机森林(Bagging)
Bagging 应用于同类型的模型个体，使其形成模型群体。通过有放回随机抽取全体数据中部分样本来训练出有差异的个体模型，然后将这些个体模型组合成模型群体来实现融合模型。Badding 随机性不但包括样本选择随机性，而且在个体模型中节点的特性也是随机选取来达到个体差异性。

### Boosting(GBDT)
在 Bagging 中模型样本和特征随机选取可能具有盲目性，也就是前面训练模型并没有影响到后序模型训练，模型之间训练相对独立。而在 Boosting 训练是有时序性的，也就是后序模型是在前面模型的训练成果继续训练的，后续模型会尝试修正前面模型的错误。主要两种分别是**Ada-Boosting**和**Gradient-Boosting**。

#### Ada-Boosting
在Ada-Boosting 训练方式是，后续模型更关注之前的错误样本，具有一定针对性训练。
- 为整个数据集每一个样本分配权重$s_1,s_2,\dots , s_m$来表示模型更关心那些样本
- 归一化序列$[acc_1,acc_2,\dots,acc_n]$
#### Gradient-Boosting
这种融合模型执行比较简洁，但数学推导以及实现相对要复杂些，可以用于概率分类和回归问题。方式是后续模型不再直接预测数据集的预测值，而是预测之前模型的预测值和真实值的差值。
- 训练模型$M_1$ 得到预测值后 $ yPred = yPred_1$
- - 下一个模型$M_2$将预测值为$dy_1 = yPred_1 - y$ 作为训练的目标，所以总体模型预测值 $yPred = yPred_1 + yPpred_2$


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



我们知道 tensorflow 是机器学习主流框架，这一点我们从机器学习岗位招聘要求就可以了解到 tensorflow 的重要。在 tensorflow 框架基础上有更高级的 keras 等框架。

我们之前使用决策树解决过这个分类问题，今天我们继续用 tensorflow 来解决问题，这个代码是从 kaggle 上借鉴过来。我们可以一起一边读一边了解作者是如何解决这个问题。

我们首先来构建图
```python
inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
labels = tf.placeholder(tf.float32, shape=[None, 1])
```

我们要做几件事
- 初始化变量
- 定义神经网络
- 定义目标函数(损失函数)
- 优化损失函数找到最优参数，也就是找到
#### 初始化变量
```python
inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
labels = tf.placeholder(tf.float32, shape=[None, 1])
learning_rate = tf.placeholder(tf.float32)
```
输入[None,特征值个数]的向量，这里因为输入样本数量不确定，所以使用 None。
输出为[None,1] 的一列矩阵，最后定义一下学习率。
#### 定义神经网络
```python
initializer = tf.contrib.layers.xavier_initializer()
fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
fc=tf.layers.batch_normalization(fc, training=is_training)
fc=tf.nn.relu(fc)
 logits = tf.layers.dense(fc, 1, activation=None)
```
先定义一个全连接层，输入为 inputs 输出为 hidden_units 个节点
激活函数 relu，这个激活函数大多数神经网络层激活函数，而不是 sigmoid 的激活函数。
#### 定义目标函数(损失函数)
```python
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
cost = tf.reduce_mean(cross_entropy)
```
这里使用sigmod的交叉熵作为损失函数


```python
    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```
使用逻辑回归来进行分类，分类结果 predicted 和 labels 进行对比来进行

```python
export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

```

```python
epochs = 200
train_collect = 50
train_print=train_collect*2

learning_rate_value = 0.001
batch_size=16
```
上边的参数如果学习过 tensorflow 应该不会陌生，epochs 表示训练的次数，batch_size 表示每一个训练批次样本数量，learning_rate_value 表示学习率。

```
with tf.Session() as sess:
```
设计好的计算图都需要放到 tf.Session() 进行计算
```
sess.run(tf.global_variables_initializer())
```

```python
saver = tf.train.Saver()
saver.save(sess, "./titanic.ckpt")
```
将训练模型参数保存起来
