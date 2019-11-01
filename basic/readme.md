



### 多元线性回归
之前我们完整地介绍了什么是一元线性回归问题，以及通过梯度下降来求解到损失函数最小时所对应的参数。从而找到一元线性回归的最优解。今天我们将问题扩展到多元回归问题。在回归分析中，如果有两个或两个以上的**自变量**，就称为多元回归。
在回归分析中，如果有两个或两个以上的**自变量**，就称为多元回归。
多元线性回归与一元线性回归类似，可以用**最小二乘法**估计模型参数，也需对模型及模型参数进行统计检验。
### 多元线性回归模型
$$h(\theta_1,theta_2, \cdots \theta_m) = \theta_0 + \theta_1x_1  \theta_2x_2 + \cdots \theta_mx_m$$
之前我们了解到了多元线性回归是用线性的关系来拟合一个事情的发生规律,找到这个规律的表达公式,将得到的数据带入公式以用来实现预测的目的,我们习惯将这类预测未来的问题称作回归问题.机器学习中按照目的不同可以分为两大类:回归和分类.今天我们一起讨论的逻辑回归就可以用来完成分类任务.
$$ g(x) = \frac{1}{1 + e^{-x}} $$

sigmoid

$$ h_{\theta}(x) = \frac{1}{1 + e^{-\theta^TX}} $$

$$ h_{\theta}(x) = \theta_0 + \theta_1 x$$

$h$ 表示函数 $\theta_0$ 表示截距(以后叫偏移值) $\theta_1$ 表示斜率（以后叫权重)

$$ h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_n x_n$$ 


多元线性回归
$$ h_{\theta} = \theta^Tx= \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_n x_n$$
这里解释一下参数其中 $\theta_0,\theta_1 \dots \theta_n $ 这里$\theta^T$是向量转置，也就是行列变换，有 n 行 1 列向量变换为 1 行 n 列。
接下里看一下他的损失函数，没有什么不同就是在一元线性回归基础
$$ J(\theta_0,\theta_1 \dots \theta_n) = \frac{1}{2m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2$$
接下来就是我们通过梯度下降法来优化我们参数来达到损失函数最小值
$$ \theta_j := \theta_j - \eta \frac{\partial}{\partial \theta_j} J(\theta_0,\theta_1 \dots \theta_n)$$
分类准确率
最大似然估计法(maxiumu likelihood)
**似然性**就是概率，也就是我们假设的使我们数据最大，也就是在这个假设下数据出现的最大。

$$P(D|\theta)$$
$$P((x_1,y_1),(x_2,y_1),\dots,(x_n,y_n)|\theta)$$
联合分布也就是所有数据出现的可能形式
$$P((x_1,x_2,\dots,x_n),(y_1,y_2,\dots,y_n)|\theta) $$
$$P(\vec{Y},\vec{X} | \theta) $$
### 离散型随机变量
设总体 $X$ 是离散型随机变量，概率分布为 $P\{ X = t_i \} = p(t_i;\theta),i = 1,2, \dots$ 其中 $\theta \in \Phi$ 为带估计参数。
设$X_1,X_2, \dots ,X_n$ 是来自总体样本 X 的样本，$x_1,x_2, \dots,x_n$ 是样本值，成函数$ L(\theta) = L(x_1,x_2, \cdots , x_n;\theta) = \prod_{i=1}^n p(x_i;\theta) $ 为样本 $x_1,x_2, \dots x_n$ 的似然函数，如果$\theta \in \Phi$ 使得 $L(\hat{\theta}) = \max_{\hat{\theta} \in \Phi} L(\theta)$ 这样 $\hat{\theta}$ 与 $x_1,x_2, \dots,x_n$  有关，记作$\hat{\theta}(x_1,x_2, \dots,x_n )$,称未知参数 $\theta$ 的最大似然估计值，相应的统计量$\hat{\theta}(x_1,x_2, \dots,x_n )$ 称为 $\theta$ 的最大似然估计量。
### 连续型随机变量
设总体 $X$ 的概率密度函数 $f\{ X = t_i \} = p(t_i;\theta),i = 1,2, \dots$ 其中 $\theta \in \Phi$ 为带估计参数。
设$X_1,X_2, \dots ,X_n$ 是来自总体样本 X 的样本，$x_1,x_2, \dots,x_n$ 是样本值，成函数$ L(\theta) = L(x_1,x_2, \cdots , x_n;\theta) = \prod_{i=1}^n p(x_i;\theta) $ 为样本 $x_1,x_2, \dots x_n$ 的似然函数，如果$\theta \in \Phi$ 使得 $L(\hat{\theta}) = \max_{\hat{\theta} \in \Phi} L(\theta)$ 这样 $\hat{\theta}$ 与 $x_1,x_2, \dots,x_n$  有关，记作$\hat{\theta}(x_1,x_2, \dots,x_n )$,称未知参数 $\theta$ 的最大似然估计值，相应的统计量$\hat{\theta}(x_1,x_2, \dots,x_n )$ 称为 $\theta$ 的最大似然估计量。

我们之前学习回归函数$h_{\theta}(\theta^Tx)$ 样子想必大家已经都熟悉了吧。$g(x) = \frac{1}{1+e^{-x}}$
我们把$h_{\theta}$作为变量带入g(x) 中就得到今天逻辑回归的模型
$$ h_{\theta} = \frac{1}{1 + e^{- \theta^TX }} $$
我们之前学习回归函数$h_{\theta}(\theta^Tx)$ 样子想必大家已经都熟悉了吧。$g(x) = \frac{1}{1+e^{-x}}$
我们把$h_{\theta}$作为变量带入g(x) 中就得到今天逻辑回归的模型
$$ h_{\theta} = \frac{1}{1 + e^{- \theta^TX }} $$

我们已经知道逻辑回归是用来做分类的，一般分为 2 个 类别，我们将 0.5 作为分类边界。sigmoid 函数特点就是 0 到 1 我们就可以小于 0.5 归为一个类别。
- 当$z\ge0$ 时候$g(z) \ge 0.5$ $z\ge0$ 也就是 $\theta^TX \ge 0$ 所以这时$g(\theta^TX) \ge 0.5$
- 当$z\le0$ 时候$g(z) \ge 0.5$ $z\le0$ 也就是 $\theta^TX \ge 0$ 所以这时$g(\theta^TX) \le 0.5$

### 决策边界
$$ h_{\theta}(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_2^2 + \theta_4x_2^2  ) $$

### 损失函数

$$ J(h_{\theta}(x),y) = \begin{cases}
  -\log(h_{\theta}(x)) & y = 1 \\
  -\log(1 - h_{\theta}(x)) & y = 0
\end{cases} $$

$h_{\theta}(x)$接近于 1 时候**损失函数**就为 0
我们是无法对分段函数进行求导，不过这里用了一个小技巧来解决这个问题。
$$ J(h_{\theta}(x),y) = -y\log(h_{\theta}(x)) - (1-y)\log(1 - h_{\theta}(x))$$
- 当$y=1$时，$J(h_{\theta}(x),y) = -\log(h_{\theta}(x)) $
- 当$y=0$时，$J(h_{\theta}(x),y) = -\log(1 - h_{\theta}(x))$
$$
\begin{cases}
  \theta_0 = \theta_0 - \eta \frac{1}{m} \sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)}) \\
  \theta_1 = \theta_1 - \eta \frac{1}{m} \sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}
\end{cases}
$$

### 梯度下降
$$ J(\theta) = -\frac{1}{m}[\sum_{i=1}^m y^{(i)}\log h_{\theta}(x^{(i)}) + (1-y^{(i)})\log(1-h_{\theta})\log(1-h_{\theta}(x^{(i)})) ]$$

$$ 2x \rightarrow 2 $$
$$ x^2 \rightarrow 2x $$
$$ \frac{\partial g(f(x))}{\partial}  \rightarrow \frac{\partial g(f(x))}{\partial f(x)} \frac{\partial f(x)}{\partial x} $$

$$ \frac{1}{2m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2 \rightarrow 2 \cdot \frac{1}{2m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) \rightarrow \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) $$

$$ \frac{1}{2m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2 \rightarrow 2 \cdot \frac{1}{2m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) \rightarrow \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$


### 无监督学习和监督学习
我们现在研究大多数都是

$$ \frac{1}{1 + e^{x}} $$

线性回归(linear regression)是用来做回归(预测)的，逻辑回归(logistic regression)是用来做分类的。虽然两者用于

解决不同问题，但是因为名字里都有回归所以还是有着一定关系不然就叫逻辑分类。
### 伯努利分布
伯努利分布(Bernoulli distribution)又名两点分布或 0-1 分布，在介绍伯努利分布前先介绍一下**伯努利实验**
**伯努利试验**是只有两种可能结果的单次随机试验，即对于一个随机变量X而言。
进行一次伯努利试验，成功(X=1)概率为p(0<=p<=1)，失败(X=0)概率为1-p，则称随机变量X服从伯努利分布。
$$P(y=1|x,\theta) = h_{\theta}$$
$$P(y=0|x,\theta) = 1 - h_{\theta}$$
### 最大似然估计
最大似然估计的意思就是最大可能性估计,其内容为:如果两件事 A,B 相互独立,那么A和B同时发生的概率满足公式
$$ P(A,B) = P(A) \cdot P(B)$$


### 训练集和测试集
### 梯度下降
梯度下降是迭代法的一种,可以用于求解最小二乘问题(线性和非线性都可以)。在求解机器学习算法的模型参数，即无约束优化问题时，梯度下降(Gradient Descent)是最常采用的方法之一，另一种常用的方法是最小二乘法。除了梯度下降我们随后还会介绍其他优化算法。不多神经网络用的。
梯度下降是一种优化算法，
$$ w_{(i+1)} = w_i - \eta \frac{\partial L(w)}{ \partial w}$$
我们初始化参数 w 然后按某个方向在损失函数移动 w 移动距离是由损失函数对 w 导数和一个学习率的乘积。如果变化率小于 0 。

问题
局部最小值

学习率使用，
- 如果学习率设置过小，每次只前进一小步这样缓慢下去可能求失去耐心了
- 如果学习率设置过多，我们可能一步就越过谷底，在谷底附近进行震荡而始终无法达到谷底。
- 随后我们会根据训练步骤以及不同参数不断调整学习率，现在只要知道学习率是比较关键的一个参数即可。以后会不断更新我们对学习率认识。

### 概率派和贝叶斯派(Bayesian)
- 数据集 $$ X:  {x^1, x^2 \cdots x^i }$$
- 参数 $$ \theta $$


### 统计学研究方向
什么是统计是关于数据的学科，我们想要增加对大自然了解。搜集数据然后对自然进行推断。我们日常计算均值和方程并不是统计学研究问题，统计学主要研究统计推断。统计学主要两个学派——频率学派和贝叶斯学派
- 频率学派
知道参数的点估计，然后再告诉这个点估计到底有多大准确度。
- 贝叶斯学派
贝叶斯脱离统计的，统计模型、线性模型或logist模型，对于模型参数进行进行预测，在贝叶斯得到分布，有一个先验条件，在贝叶斯对参数进行推断然后得到参数的分布。
我们假设一个样本是根据工作岗位的条件一名程序员是否接收工作的

| 接收offer | 常出差 | 加班 |  高工资 |
| ------ | ------ | ------ | ------ |
| 0 | 1 | 1 |0 |
| 1 | 0 | 0 |1 |
| 1 | 1 | 0 |1 |
| 0 | 0 | 1 |0 |
| 0 | 0 | 1 |1 |
| 1 | 0 | 1 |1 |

- A  表示接受offer  
- B 表示常出差 
- C 表示加班 
- D 表示高工资
如果一个人接收一个岗位（没有加班，没有加班高工资的岗位）接收概率。
从样本来看 $ P(A=1) = \frac{1}{2} $ 表示接受 offer 可能性为 $ P(A=0) = \frac{1}{2}$
我们在看一看常加班概率 $P(B=1) = \frac{1}{3} $ 和 $P(B=0) = \frac{2}{3}$
$P(A\bigcap B) = \frac{1}{6}$
在条件概率也就是 B已经发生了 A发生概率用 $P(A|B)$ 来表示。在已经知道加班的条件接收offer的概率，看表是 $\frac{1}{2}$
我们看已知加班接收offer概率是 $\frac{1}{4}$。说明大多数人都讨厌加班。
然后我们看看高工资条件下接收 offer 概率是$\frac{3}{4}$

$$ P(A|B) = \frac{P(A \bigcap B)}{P(B)} $$
$$ P(A|B)P(B) = P(A \bigcap B) $$

如果已知 A 然后 B 的概率
$$ P(B|A) = \frac{P(A \bigcap B)}{P(A)} $$
$$ P(B|A)P(A) = P(A \bigcap B) $$
然后我们将上面两个公式进行变换后消去相同的部分

$$ P(A|B)P(B) = P(B|A)P(A)$$
最终我们就可以得到这个公式，这个公式就是贝叶斯公式。
$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

$ y = f(x)$
- y ： 接收 offer
- x : $x_1$ 表示常出差 $x_2$ 表示常加班 $x_3$ 表示高工资
$$ P(y=1|x_1,x_2,x_3) $$
$$ P(y=0|x_1,x_2,x_3) $$
$$ \begin{cases}
    x_1 = 0 \\
    x_2 = 0 \\
    x_3 = 1 \\
\end{cases} $$
最终我们将这些参数带入到贝叶斯公式
$$ P(y|x_1,x_2,x_3) = \frac{P(x_1x_2x_3|y)P(y)}{P(x_1x_2x_3)} $$
我们无需看分母，因为无论 y = 1 或者 y = 0 他们分母都是一样的。
$P(x_1x_2x_3|y)$ 这部分内容看起来比较难算，可以马可夫进行

$$ P(y|x_1,x_2,x_3) \approx P(x_1|y)P(x_2|y)P(x_3|y)P(y) $$

$$ P(y=1|x_1,x_2,x_3) \approx P(x_1|y)P(x_2|y)P(x_3|y)P(y) $$

$$ P(y=1|x_1,x_2,x_3) \approx P(0|1)P(0|1)P(1|1)P(y) $$

- 接收offer没有出差
$P(0|1)$ 表示已知接收offer了有多少没有出差情况，可以查表得到接收offer没有出差是 $\frac{2}{3}$

- 接收offer没有加班
$P(0|1)$ 表示已知接收offer了有多少没有加班情况，可以查表得到接收offer没有加班是 $\frac{2}{3}$

- 接收offer高薪
$P(0|1)$ 表示已知接收offer了有多少高薪情况，可以查表得到接收offer高薪是 1

- 接收offer概率
  概率是 $\frac{1}{2}$

那么我么就很轻松求取概率
不加班不出差高薪招聘到人的概率
$$ P(y=1|x_1,x_2,x_3) \approx \frac{2}{3}\frac{2}{3} \frac{1}{2} \approx 22%$$

不加班不出差高薪招聘不到人的概率
$$ P(y=0|x_1,x_2,x_3) \approx \frac{2}{3}\frac{2}{3} \frac{1}{3} \frac{1}{2} \approx 7.3%$$

从结果来看，不加班，不出差高薪还是很容易招聘到人


$$ P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} $$
### 什么是贝叶斯统计

### 拉普拉斯
### 傅里叶
### 拉格朗日
## 假设验证
我们无法对全体样本进行调查，所有需要抽样来推测总体样本的情况
### 概率基础知识
假设标准正态分布 $u=0, \sigma=1$ 
<!-- x 小于等于 1 大于等于 -1 $P$ -->
落在 1 倍标准差内的概率
$$ P(-1 \le x \le 1 ) = 68 \% $$ 
$$ P(-2 \le x \le 2 ) = 95 \% $$
$$ P(-3 \le x \le 3 ) = 99.73 \% $$

###