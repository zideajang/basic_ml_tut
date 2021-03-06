## 最近邻规则分类

**KNN**(K-Nearest Neighbours)K值邻近算法, 是一个简单的, 常被用于分类问题的算法, 也可以用于**回归问题**。KNN 是**非参数**的,基于实例的算法。

在统计学中，**参数模型**通常是假设总体服从某个分布，这个分布可以由一些参数确定，如正态分布由**均值**和**标准差**确定，在此基础上构建的模型称为参数模型

常见的参数机器学习模型有：
1、逻辑回归（logistic regression）
2、线性成分分析（linear regression）
3、感知机（perceptron）

**非参数模型**对于总体的分布不做任何假设或者说是数据分布假设自由，只知道其分布是存在的，所以就无法得到其分布的相关参数，只能通过非参数统计的方法进行推断。

常见的非参数机器学习模型有：
1、决策树
2、朴素贝叶斯
3、支持向量机
4、神经网络

其中最常见的计算各点之间的方法是**欧氏距离**(Euclidean Distance)。欧氏距离就是计算 N 维空间中两点之间的距离。

$$ dist = \sqrt{\sum_{i=1}^N (x_i - x_i)^2}$$

```python
  buying  maint door persons lug_boot safety  class
0  vhigh  vhigh    2       2    small    low  unacc
1  vhigh  vhigh    2       2    small    med  unacc
2  vhigh  vhigh    2       2    small   high  unacc
3  vhigh  vhigh    2       2      med    low  unacc
4  vhigh  vhigh    2       2      med    med  unacc
```


KNN 模型(K-Nearest Neighbor)
KNN 算法的一个最吸引人的特点是简单易懂，易于实现在零到很少的训练时间，可以成为一个有用的工具，对一些数据集的即时分析，这也是KNN 的优点。因为 KNN 不需要带有参数的模型进行训练，所以 KNN 是无参数的模型，可以用于分类和回归。其中 K 表示在新样本点附件选取 K 个样本数据，通过在 K 个样本进行投票来判断新增样本的类型。这里值得注意 k 并不是模型参数而是我们事先指定的

- k 是否可以取偶数，当 K 为偶数时，如果 K 点不同类别样本点各占一半，我们不是无法对新样板点进行推测。我们可以通过距离来进一步推测样本点类别。

$$ w_i = \frac{1}{d_i}$$
- k 取 1 可能发生过拟合，如果 K 选取无穷大，那么就变成哪个点多就把任何新的数据分为哪个数据多类别
- 任何两个点距离的之间连接连线，每一个小块，最近邻的数据，不再是中线而是面，
- k近邻可以做预测，估计一个人成绩，也就是我们可以通过临近点的均值或者带有按离推测样本点距离作为权重来计算该样本点的值。

#### A 和 B 两点间距离需要满足以下条件

- 对称 d(A,B) = d(B,A) 也就是 A 到 B 的距离等于 B 到 A 的距离
- d(A,A) = 0 样本点到自己本身的距离为 0 
- d(A,B) = 0 iff A = B 两点距离为 0 那么这两个点就是同一个点
- d(A,B) <= d(A,C) + d(B,C) 三角不等式

#### 距离(模)
- 欧氏距离
$$ dist(x,y) = \sqrt{{\sum_{i=1}^d (x_i - y_i)^r}} = ||x - y ||_2$$
- 曼哈顿距离
$$ dist(x,y) = {\sum_{i=1}^d |x_i - y_i|} = ||x -y|| $$
- 切比雪夫距离
$$ dist(x,y) = ({\sum_{i=1}^d |x_i - y_i|^r})^{-r} $$
- hamming distance(异或运算)，通过两个二进制数做异或来得到估计值
- consine similarity
$$\cos(x,y) = \frac{x^Ty}{||x||_2 ||y||_2}$$

### 特征工程
| 种类  |   |
|---|---|
|   |   |
