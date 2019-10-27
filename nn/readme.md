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
