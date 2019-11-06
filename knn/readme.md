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


赞成的意见。正如您已经从上一节中了解到的，K近邻算法的一个最吸引人的特点是它简单易懂，易于实现在零到很少的训练时间，它可以成为一个有用的工具，对一些数据集的即时分析，你正计划运行更复杂的算法此外，KNN在处理多类数据集时同样容易，而其他算法则是针对二进制设置进行硬编码的最后，正如我们前面提到的，KNN的非参数特性在某些数据可能非常“不寻常”的情况下给了它一个边。