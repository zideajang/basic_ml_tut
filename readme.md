## 机器学习基础篇

## 特征缩放和交叉验证法
$$x_1 = 1000000 cm^2 - 2000000 cm^2 \tag{area}$$
$$x_2 = 1 - 5 \tag{room count}$$
因为特征间数据取值范围相差过大，就会造成梯度下降会走的很远。这样优化时间比较长而且可能造成错误路径。
### 数据归一化
- 就是把数据的取值范围处理为 0 - 1 或者 -1 1 之间
    - 任意数据转化为 0 - 1 之间 ( $ newValue = \frac{oldValue - min}{max - min}) $
    - 任意数据转化为 -1 - 1 之间  ( $ newValue = \frac{oldValue - min}{((max - min) - 0.5) * 2} )$
### 均值标准化
- x 为特征数据，u 为数据的平均值，s 为数据的方差
$$ newValue = \frac{oldValue - u}{s}$$
- 取值范围从 -0.5 - 0.5

### 交叉验证法
- 通常我们会将数据集按一定比例进行切分为训练数据集和测试数据集
- 对于较小数据集时候我们就会用到交叉验证法
#### 交叉验证法做法
- 把所有的数据切分为 10 份，如果有 100 样本切分每个数据集有 10 个样本

$$ E = \frac{1}{10} \sum_{i=1}^{10} E_i $$
## 决策树算法
### 决策树的概率
### 决策树的算法
#### 纯度计算公式
#### 从 ID3 到 C4.5
#### 问题
```python
1. 1 1 => 1
2. 1 2 => 1
3. 0 0 => 0
4. 0 0 => 0
5. 1 1 => 1
6. 0 0 => 0
7. 1 2 => 1
8. 1 1 => 1
9. 0 2 => 0
10. 1 0 => 0

```

- 熵概念: 信息熵是用来描述信息的混乱程度或者信息的不确定度。信息熵是可以反映集合的纯度
- 算法目的通过不断决策树来将信息熵降低下来
- 信息熵 $$ Ent(D) = \sum_{k=1}^m p_k \log_2 \frac{1}{p_k} = - \sum_{k=1}^m p_k \log_2 p_k$$
$$ p_1 = \frac{1}{2}, p_2 = \frac{1}{2}$$
$$ - ( \frac{1}{2} \log_2 \frac{1}{2} +  \frac{1}{2} \log_2 \frac{1}{2}) = 1$$
- 信息增益 $$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

- 信息增益率
$$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$
划分条件为待遇
$$ Ent(D_1) = - (\frac{5}{6} \log \frac{5}{6} + \frac{1}{6} \log \frac{1}{6} ) \approx 0.65$$

$$ Ent(D_2) = 1 * \log_2^1 + 0 * \log_2^0 = 0 $$

$$ 1 - (\frac{6}{10} * 0.65 + \frac{4}{10} * 0) = 0.61 $$
表示划分后信息熵为，信息增益越大越好，使用信息熵天生对多路分类进行偏好
- 信息增益率
$$ Gain_ratio(D,C) = \frac{Gain(D,C)}{Ent(C)} $$

$$ Ent(C) = - \sum_{i=1}^k \frac{N(D_i)}{N} \log_2 \frac{N(D_i)}{N} $$

对于多路划分乘以代价
$$ Gain(D C_1) = 0.61, Gain(D C_2) = 0.72$$
$$ Ent(C_1) = - (\frac{6}{10} \log \frac{6}{10} + \frac{4}{10} \log \frac{4}{10}) = = 0.971$$
$$ Ent(C_2) = 1.571$$

$$ Gain_ratio = 0.63  $$

#### CART 算法
CART 即分类与回归树(Classfication And Regression Tree) CART 算法与 ID3 算法思路是相似的，但是在具体实现和场景上略有不同。
- CART 不仅可以处理分类问题还可以处理连续问题
- CART 算法采用基尼指数(分类树)以及方差（会归树）作为纯度的度量，而 ID3 系列算法采用信息熵作为纯度
- CART 算法只能二叉树
##### 分类树
$$ Gini(D) = \sum_{k=1}^m p_k(1 - p_k) = 1 - \sum_{k=1}^m p_k^2$$

$$ \Delta Gini = Gini(D) -   \sum_{i=1}^2 \frac{N(D_i)}{N} Gini(D_i)$$
##### 会归树

## 聚类算法
### 聚类算法 K-Means 算法

## SVM
### 概念 SVM (Support Vector Machine)
SVM 用于解决分问题的分类器。介绍且背后算法，了解算法后便于我们更好调整参数。

#### 参数分析
- c 用来调整松弛变量系数
- kernel 核函数用进行判别类别的函数 例如 rbf
- degree
- gamma 是关于 rbf 的特有参数
- probability 学习统计同学比较了解，结论的可能性，通过统计理论以及
- cache_size 对 SVM 计算缓存大小
- class_weight=None 每一个类别比较平均就不需要设置这个参数，样本不均匀时候是不会用到。

### SVM 分类
#### 线性
#### rbf 

### 什么是 SVM 算法
- 二元线性分类问题(简单)
    - 可分问题
    - 什么样线性方程是最好线性的方程，离这条子线最近那些点离这条线最远，这也是 SVM 的目标
    - 有很多判别线
    - 支持向量与我们直线最近那些点(向量)就是支持向量

$$  f(x) = W^Tx + b $$
$$ V^i = \frac{W^T}{||W||} x^i + \frac{b}{||W||}$$
$$ V^i = \frac{W^T}{||W||} x^i + \frac{b}{||W||}$$

## 神经网络
### 人工神经网
我们人类从自然中学习到很多，例如模仿鸟儿的飞机、模仿鱼儿的潜水艇等等。这些都属于仿生学。
神经网络也是模仿人脑的神经元，虽然计算机在一些简单问题上是完胜人脑。现在在一些复杂信息处理（视觉和语言方面）到现在人脑要远远优于计算机。

人脑神经元结构
- 树突：接收信息
- 细胞核: 在细胞核对树突收集信息进行加工处理
- 轴突和突触：处理好信息通过突触向后传递信息

计算机仿生神经元
- 信息输入$ \{x_1 x_2 \cdots x_k \} $
- 信息处理(加法器和激活函数) $\sum_{j=1}^k w_jx_j + \theta$ 和 f(x)
- 信息输出 $ y = f(\sum_{j=1}^k w_jx_j + \theta) $

#### 常规神经网的结构
- 输入层:注意除了 $ \{x_1 x_2 \cdots x_k \} $ 自变量还有一个常数项 $\theta$
- 隐藏层：没有隐藏层神经网络就是随后介绍感知机
- 输出层:输出层根据解决稳定


### 感知机(单层神经网络)
输入层 $\{ x_1 x_2 \theta \}$
输出层:
$$ sign(x) $$

#### 初始化权重
- 初始化权重
- 设计模型
- 损失函数
$$ E = \frac{1}{2}(y - \hat{y})^2$$
通过推导我们可以得到 $\Delta W_j$ 变化率的公式
$$ \frac{\partial E}{\partial W_j} = -(y - \hat{y}) x_j $$
- 优化更新权重
$$ W_j^{(t+1)} = W_j^{(t)} + \Delta W_j$$
$$ \Delta W_j = \eta(y - \hat{y}) x_j $$

#### 基本流程
1. 输入
    - 数据集 D
    - 学习率 $\eta,\eta \in (0,1]$
    - 停止条件: 误差率指定阈值 $\epsilon$ 和最大迭代次数 $T^k$
2. 初始化连接权重
3. 输入样本$ (x_{i1} x_{i2} \cdots x_{im}, \hat{y}_i )$
4. 更新权重 
$$ \Delta w = \eta(y_i - \hat{y}_i)x_i $$
$$ W^{(t+1)} = W^T + \Delta W $$
$$ \Delta \theta = \eta (y_i - \hat{y}_i) $$
$$ \theta^{(T_1)} = \theta^T + \Delta \theta $$

5. 停止条件，判断是否满足条件如果满足条件，模型误差小于指定误差阈值或是迭代次数大于最大迭代次数
6. 输出，如果满足条件就将模型作为输出
$$ y = f(\sum_{j=1}^k w_jx_j + \theta)  $$

### BP 神经网络(多层神经网络)
反向传播算法，BP(Back Propagation) 简称 BP 算法，是建立在梯度下降算法基础上适合多层神经网络的参数训练方法。
由于隐藏层节点的预测值无法直接计算，因此，反向传播算法直接利用输出层节点的预测误差反向估计上一层隐藏节点的预测误差，也就是从后往前逐层从输出层把误差反向传播到输入层，从而实现链接权重调整，这就是反向传播。
#### 为什么我们需要 BP 神经网络
- 尽管感知机可以区分线性可分数据，但是由于网络中只有两层神经元，当面对非线性可分问题，就显得无能为力。
- 通过在神经网络中添加隐藏层，解决更复杂的问题
#### BP 推导算法
$$ h(x_{h1} x_{h2} \cdots x_{hm})$$
这里将 y 假设为 m 输出表示我们推导具有一般性
$$ (y_{h1} y_{h2} \cdots y_{hm})$$
$$ W_{ij} $$
下面假设 $y_k$ 的输入为 $\beta_k$
$$ \beta_k = \sum_{j=1}^d (W_{jk} Z_j + \theta_k)$$
计算损失函数
$$E_h = \frac{1}{2} \sum_{k=1}(\hat{y}_{hk} - y_{hk})^2$$

$$\Delta W_{jk} = -\eta \frac{\partial E_h}{\partial W_{jk}} $$

$$ W_{jk} \rightarrow \beta_k \rightarrow \hat{y}_{hk} \rightarrow E_h $$
根据链式求导法则
$$ \frac{\partial E_h}{\partial W_{jk}} = \frac{\partial E_h}{\partial \hat{y}_{hk}} =  \frac{\partial \hat{y}_{hk}}{\partial \beta_k} = \frac{\partial \beta_k}{\partial W_{jk}} $$

$$  \frac{\partial E_h}{\partial \hat{y}_{hk}} = \hat{y}_{hk} - y_{hk} $$

$$ y(x) = \frac{1}{1 + e^{-x}}$$
$$f(x)(1-f(x)$$

$$  \frac{\partial \hat{y}_{hk}}{\partial \beta_k} =  \hat{y}_{hk} (1 - \hat{y}_{hk})$$

$$ \frac{\partial \beta_k}{\partial W_{jk}} = Z_j $$

$$ \Delta W_{ij} = - \eta (\hat{y}_{hk} - y_{hk}) \hat{y}_{hk} (1 - \hat{y}_{hk}) Z_j$$
#### BP 神经网络的学习算法
1. 输入
    - 数据集 D
    - 学习率 $\eta,\eta \in (0,1]$
    - 停止条件: 误差率指定阈值 $\epsilon$ 和最大迭代次数 $T^k$
2. 初始化连接权重
3. 输入样本$ (x_{i1} x_{i2} \cdots x_{im}, \hat{y}_i )$
4. 更新权重 
$$ \Delta w = \eta(y_i - \hat{y}_i)x_i $$
$$ W^{(t+1)} = W^T + \Delta W $$
$$ \Delta \theta = \eta (y_i - \hat{y}_i) $$
$$ \theta^{(T_1)} = \theta^T + \Delta \theta $$

5. 停止条件，判断是否满足条件如果满足条件，模型误差小于指定误差阈值或是迭代次数大于最大迭代次数
6. 输出，如果满足条件就将模型作为输出
$$ y = f(\sum_{j=1}^k w_jx_j + \theta)  $$



$$ f(x) = \frac{1}{1 + e^{-1}} $$
- 输出层
- 隐藏层

##### 反向求导
$$ \Theta $$

## 深度神经网咯
### 循环神经网
#### 什么是神经网
#### 当下流行神经
#### 什么是循环神经网
在使用 google 进行搜索时候，当我们输入要搜索内容，google 会根据我们喜欢提出自动补全的文字，帮助我们快速找到要查询的内容。
#### RNN 内部机制
#### 梯度消失和梯度爆炸
#### 长短期循环神经网
#### 实例