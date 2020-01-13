### 图模型
所谓上帝视角就是我们想要得到一个全概率分布，也就是联合概率分别P(A,B,C) 我们就可以回答各种各样的问题P(A|B) P(B) P(C|A,B)。但是问题我们是在没有任何先验知识情况下我们要对参数进行建模，我们参数是随着参数增加成指数级增加的。
$$ \{A,B,C\} $$

$$\begin{Bmatrix}
    0 & 0 & 0 \\
    0 & 0 & 1 \\
    0 & 1 & 0 \\
    0 & 1 & 1 \\
    1 & 0 & 0 \\
    \vdots & \vdots & \vdots
\end{Bmatrix}$$
$$probability = 2^3 - 1$$
当我们不知道 A、B 和 C 之间关系我们是需要 7 个数来如果我们特征增加了那么参数数量就会指数增加。为了解决这个问题我们引入图模型。


E事件依赖于C事件，而 A 或 B 事件都可以触发 C 事件 ，A 事件同时触发 C 和 D事件，其实真实世界中事件间总是有一些关系可寻的。当我们把关系定义出来之后，特征有这种图的依赖关系之后，联合分布就写成一些小的概率分布乘积。写成每一个节点在给定其父节点条件概率

$$P(A,B,C,D,E) = P(A)P(B)P(C|A,B)P(D|A)P(E|C)$$
这样就很容易写出联合概率
$$P(x_1,x_2, \dots , x_n) = \prod_{i=1}^n p(x_i | parent(x_i))$$

特征之间依赖关系需要我们自己定义，虽然可以通过学习。应用场景多数在医院，医生对病例诊断，BI 也会用到图模型。

多云可能会影响会影响是否喷水，下雨，喷水和小雨一定会影响草湿

| P(C=F)  | P(C=T)  |
|---|---|
| 0.5  | 0.5  |

| C  | P(S=F)  | P(S=T) |
|---|---|---|
| F  |  0.5 | 0.5|
| T  | 0.9  | 0.1 |

| C  | P(R=F)  | P(R=T) |
|---|---|---|
| F  |  0.8 | 0.2|
| T  | 0.2  | 0.8 |

| S| C  | P(W=F)  | P(W=T) |
|---|---|---|---|
| F | F  |  1.0 | 0.0|
| T | F  |  0.1 | 0.9|
| F | T  |  0.1 | 0.9|
| T | T  |  0.01 | 0.99|


$$fib(n) = \begin{cases}
    1 & n = 1 & or & n = 2 \\
    fib(n-1) + fib(n-2)
\end{cases}$$

$$ OPT(8) \begin{cases}
    selected & 4 + OPT(5) \\
    unselected & OPT(7)
\end{cases}$$

$$ OPT(i) \begin{cases}
    selected & V_i + OPT(prev(i)) \\
    unselected & OPT(i-1)
\end{cases}$$

| task  | prev(i)  | OPT(i) |
|---|---|---|
| prev(8)  | 5  | 13 |
| prev(7)  | 3  | 10|
| prev(6)  | 2  | 9|
| prev(5)  | 0  | 9|
| prev(4)  | 1  | 9|
| prev(3)  | 0  | 8 |
| prev(2)  | 0  | 5|
| prev(1)  | 0  | 5|
这里表示如果要 i 任务，那么在其之前最近可选任务哪个任务来就是上表要表示的，
$$ OPT(8) \begin{cases}
    OPT(7) \begin{cases}
        OPT(6) \begin{cases}
            \underbrace{OPT(5)}_{overlapping}  \\
            OPT(2) + v
        \end{cases} \\
        OPT(3) +v
    \end{cases} \\
    OPT(5) + v\\
\end{cases}$$

$$ closenessWeight(h,w) = exp(-\frac{(h - \frac{winH - 1}{2})^2 + (w - \frac{winW - 1}{2})^2}{2 \sigma^2})$$

其中
$$\begin{cases}
    0 \le h \le win \\
    0 \le w \le winW
\end{cases}$$

$$ similarityWeight(h,w) = exp(-\frac{ ||I(r,c) - I(r + (h - \frac{winH - 1}{2}),c + (w - \frac{winW - 1}{2}) ) ||^2}{2 \sigma^2})$$
$$180 \degree$$
显然每一个位置的相似性权重模板是不一样的。

最后 closenessWeight 和 similarityWeight 的对应位置相乘(即点乘)然后进行归一化

$$ I = \left( \begin{matrix}
    1 & 2 \\
    3 & 4 \\
\end{matrix} \right) $$

$$ K = \left( \begin{matrix}
    -1 & -2 \\
    2 & 1 \\
\end{matrix} \right) $$

假设我们输入矩阵 A 是 $m \times n$ 的维度，NMF 将其分解为两个矩阵 $A_dash$ 和 H。
$$A = A_{dash} \cdot H$$
将矩阵 A 维度降低为 d，也就是将$m \times n$ 分解为 $m \times d$ 其中 d 远小于 n NMF 将它转化为一个优化问题，也就是最小化函数
$$ | A - A_{dash} \cdot H|^2$$

| Movie ID  |  电影名  |
|---|---|
| 1  |  星球大战  |
| 1  |  骇客帝国  |
| 1  |  盗梦空间  |
| 1  |  星球大战  |
| 1  |  星球大战  |
| 1  |  星球大战  |
| 1  |  星球大战  |
| 1  |  星球大战  |
| 1  |  星球大战  |
