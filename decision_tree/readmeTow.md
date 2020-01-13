#### 参考资料
> Tom Mitchell machine learning
#### 决策树
- 根节点(root node)
- 分支(branch)
- 节点(node)
- 页节点(leaf)

#### 不纯度(Impurity)
homogeneous
提取
#### 信息论(information theory)
C.shannon 最早信道通讯
$$01011 \rightarrow 01011$$

我们生活的确有很多信息，X 表示要表达的信息。
如何衡量信息量
$$H(X)?$$
$$H(X_1) > H(X_2)$$
1. 太阳从东方升起
2. 获取晋升



$$\begin{cases}
    H(X) \Leftrightarrow  \frac{1}{P(X)} \\
    H(X_1,X_2) \Leftrightarrow H(X_1) + H(X_2) \\
    H(X) \ge 0 
\end{cases}$$

$$H(X) = \log \frac{1}{P(X)}$$
$$H(X_1,X_2) = \log \frac{1}{P(X_1)P(X_2)}$$
$$= - \log(P(X_1)P(X_2))$$
$$= - \log(P(X_1) + \log P(X_2))$$

$$H(X) = -\log P(X)$$

#### Entropy
$$E_x[H(X)] = -\sum_{x} P(X) \log P(X)$$

$$\begin{aligned}
    E_x[f(x)] = \sum_xP(x)f(x)
    = \int_x p(x)f(x)dx
\end{aligned}$$

封闭物理系统，从有序状态变为无序的状态。

分叉点，分类，不同类别数据交界的地方。

original entropy(X) = 
$$- \frac{1}{2} \log \frac{1}{2} - \frac{1}{2} \log \frac{1}{2} = 1$$
$$entrop(A_1) = 0$$
$$entrop(A_2) = -\frac{2}{7} \log \frac{2}{7} - \frac{5}{7} \log \frac{5}{7}$$

信息增益(IG)
$$ IG = ori-entropy - \sum_i W_i Entroy(A_i) $$

$$P(X_1) P(X_2)$$

| Day  | Outlook  | Temperatur|Humidity  | Wind  | PlayTennis|
|---|---|---|---|---|---|
|   |   |   |   |   |   |


$$\begin{aligned}
    wx_2 + b  = 1 \Rightarrow w(x_1 + \lambda w) + b = 1\\
    \Rightarrow wx_1 + \lambda w^T w + b = 1\\
    \Rightarrow \lambda w^Tw = 2\\
    \Rightarrow \lambda = \frac{2}{||w||^2}
\end{aligned}$$

$$||x_2 - x_1|| = \lambda ||w|| = ||w|| \cdot \frac{2}{||w||^2} = \frac{2}{||w||}$$

这样我们就构造间距，
- 首先我们要满足所有这些数据都被正确分类 ,而且这些数据都在我们边界之外, 用数学表达这个形式 $ y_i(wx_i + b) \ge 1$
- 在满足上面条件下让间距最大

$$\begin{cases}
    max_{w} \frac{2}{||w||} \\
    s.t. & y_i(wx_i + b) \ge 1
\end{cases}$$

我们需要找一条先不但满足可将数据，机器学习的分类问题变成一个优化问题，我们为什么需要优化，优化就是让我们模型具有更好泛化能力

$$\begin{cases}
    \min_{w}  \frac{1}{2} ||w||^2 \\
    s.t. & y_i(wx_i + b) \ge 1
\end{cases}$$

接下来就是如何优化这个有限制条件的最小值，接下来我们就开始用之前学习拉格朗日来优化这个问题

$$L(w,\alpha) = \frac{1}{2} ||w||^2 + \alpha_i (1 - y_i(w^T x^{(i)} + b)$$

$$\begin{cases}
    \frac{\partial L}{\partial w} = 0\\
    \frac{\partial L}{\partial \alpha)i} = \begin{bmatrix}
        0 \\
        0 \\
        \vdots \\
        0
    \end{bmatrix}
\end{cases}$$

难得是背后的数学，我们现在已经将最大边界问题变成一个可被优化的问题,今天我们来说一说如何把最大问题变成对偶形式，这也是SVM 中一个难点和重点。
所谓 KKT 条件就是同时兼顾不等式和等式约束条件 

#### SVM 的调参
#### 不平衡数据处理

