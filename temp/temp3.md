### 机器学习中问题
机器学习中我们一切努力都是在根据已知找到一种关系，然后利用这个找到的关系来进行估计未知。这是我们经过一段学习后对机器学习的认识。这种关系就是函数，
### 函数
函数是一种关系，这种关系可以理解为数据间的映射，映射也就是一种关系
### 向量和矩阵
### 模
#### L_0
#### L_1(曼哈顿距离)
$$||X|| = |x_1| + |x_2| + \dots |x_n|$$
#### L_2(欧式距离)
$$||X||_2 = \sqrt{|x_1|^2 + |x_2|^2 + \dots |x_n|} $$

### 向量
我们需要知道向量用来表示每一个样本的特征向量，在机器学习中我们通常用大小字母表示矩阵而用小写的字母表示 x 样本，通过上标来表示第几个样本例如$x^{(i)}$ 表示第 i 个样本，而用下标表示 x 样本的第j个特性$x_j$，这里都是机器学习中经常出现的。


### 流形学习概述

**流形学习**的观点：认为我们所能观察到的数据实际上是由一个低维流形映射到高维空间的。我们怎么理解什么是流形，我们这里解释一下。例如我们在三维空间上一个球面，这里说的是球面而不是球体，球面就是流形。球面也是一个流形，欧式空间是流形的一种特殊情况。

我们将高维空间一个平滑低维的曲面。例如我们打开20x20图片，其实图片为400维的数据，这是在 400 维的高度空间的。但是许多维度是冗余的，只有在一部分维度是有意义。

我们举一个例子，我们有时候如何认知一个陌人生呢?有时候我们需要通过他朋友和同事的信息来推测这个人。周围同事举止言谈就能反映你要认识陌生人，然后我们可以通过同事待遇加上权重来推测个陌人生的待遇。

线性组合来推断一个陌生人的生活水平。

理科平均成绩，对于数理化成绩都有 0.8 权重，而降维到理科成绩后权重被保留。

$$AX=Y$$

LLE(local linear Embedding) 局部嵌入的问题

### 推导公式
$$E(W) = \sum_i |x_i - \sum_j W_{ij}x_j|^2$$
- $x_i$ 表示第 i 个样本
- 我们通过邻接的 j 样本进行加权 $w_{ij}$ 求和来推测 $x_i$
- 所谓E(w)我们通过计算差值最小

$$\begin{aligned}
    \epsilon^{(i)} = |x^{(i)} - \sum_j w^{(j)} \eta^{(j)}|^2 \\
    = |\sum_{j=1}^k(x^{(i)} - \eta^{(j)}) \cdot w^{(j)}|^2
    \end{aligned}$$

$x_i$ 通过与邻近样本 $\eta$ 分别乘以其权重来表示，注意这里权重和需要为 1。我们注意上面公式推导，将$x^{(i)}$ 从加和符号外部移动到了加和符号$\sum$ 内部。这个公式是怎么推导出，这里可以演示一下推导过程
$$ \begin{aligned}
\Rightarrow \sum_j(x^{(i)} - \eta^{(j)} )  w^{(j)}  \\
\Rightarrow \sum_j(x^{(i)}w^{(j)} - \eta^{(j)}w^{(j)} )  \\
\Rightarrow \sum_jx^{(i)}w^{(j)} -\sum_j \eta^{(j)}w^{(j)}  
\end{aligned}$$

我们知道$\sum_j w_j$ 是为1 所以上面推导出式子就等价于
$$x^{(i)} -\sum_j \eta^{(j)}w^{(j)}  $$

$$ X =  \begin{bmatrix} 
    x^{i} - \eta^1 \\
    x^i - \eta^2 \\
    \vdots \\
    x^i - \eta^j
    \end{bmatrix}$$

$$ W = \begin{bmatrix}
    w^{(1)} \\
    w^{(2)} \\
    \vdots \\
    w^{(k)} \\
\end{bmatrix}$$

因为W 是一个 j 向量而可以将 j 个 $x^i - \eta^i$ 组成向量记做 X 这样就可以将损失函数写成向量的表达形式。

$$\epsilon^i = |XW|^2$$

$$\begin{aligned}
    \epsilon^{(i)} = |XW|^2 \\
    = (XW)^T XW \\
    = W^TX^TXW
\end{aligned}$$

这里推导过程中对于大家来说理解可能有问题就是就是$(XW)^T \Rightarrow W^TX^T$ 也就是分别进行转置时候需要调换位置，可以复习一下相应的线性代数知识。

$$I^TW = 1$$
因为W是一个权重和唯一向量所以我们可以乘以单位向量转置来表示权重加和为 1 的形式。也就是等价于$\sum_j w^{(j)} = 1$

$$\begin{cases}
    min \epsilon^{(i)} \\
    s.t. \sum_j w_i^{(j)} = 1 \Leftrightarrow I^W = 1 \Rightarrow 1 - I^TW = 0
\end{cases}$$

这种形式相比大家已经很熟悉了，也就是在等式约束条件下求最小值问题，我们可以通过拉格朗日来解决。


$$\begin{cases}
    \Rightarrow 2x^Txw =\lambda I\\
    \Rightarrow w = \frac{\lambda I}{2x^Tx} \\
    \Rightarrow w = \frac{\lambda}{2} (x^Tx)^{-1}I
\end{cases}$$

$$\begin{cases}
    \frac{\partial L}{\lambda} = 0 \Rightarrow 1 - I^Tw = 0 \Rightarrow I^Tw = 1
\end{cases}$$