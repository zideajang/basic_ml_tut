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
