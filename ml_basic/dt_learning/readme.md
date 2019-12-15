#### 参考资料
> Tom Mitchell machine learning
#### 决策树
我们根据数据特征作为节点对数据进行分类，该特征的可能值为该分类的值，每一条分支都是一个branch(分支)，每一个分支下有特征节点或者是叶子节点，如果是叶子节点就不可再分了。
- 根节点(root node)
- 分支(branch)
- 节点(node)
- 页节点(leaf)

决策树就是将不确定性比较高数据在叶子节点变得确定性很高。

#### 不纯度(Impurity)
homogeneous
提取
#### 信息论(information theory)
C.shannon 最早信道通讯，今天我们收听广播、看到电视节目和打电话无疑不是建立在通讯信息基础之上的。
$$01011 \rightarrow 01011$$

我们生活的确有很多信息，X 表示要表达的信息的数据。如何衡数据的量信息量$H(X)?$也就是如何判断两个数据量信息量的大小$H(X_1) > H(X_2)$ 
下面通过两端文字信息对比来说明一下信息量和概率之间关系。
1. 太阳从东方升起
2. 在单位获取晋升的通知

信息量和能跟事件出现的概率成反比，太阳从东方升起的消息对我们是没有什么的，因为太阳每天都会从东方升起，也就是这条消息信息量很少，而当领导告诉今年升职名单中有你的时候，这个信息量就很大了，因为这个对于你可能是小概率事件。


$$\begin{cases}
    H(X) \Leftrightarrow  \frac{1}{P(X)} \\
    H(X_1,X_2) \Leftrightarrow H(X_1) + H(X_2) \\
    H(X) \ge 0 
\end{cases}$$

通过上面的例子我们了解到概率小事件信息量反而大，而信息量大事件概率反而小。
然后就是两件事件的信息量的可能是两个事件的信息量之和。还有就是一个信息量的函数应该是大于0，有时候我们接受到信息量，即使信息量对于我们没有任何帮助，这件事也不会减少我们信息量。

所以我们构造函数H(x) 需要满足上面 3 个关系。 

$$H(X) = \log \frac{1}{P(X)}$$
信息熵
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

封闭物理系统，从有序状态逐渐变为无序的状态，分叉点，分类，不同类别数据交界的地方。

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