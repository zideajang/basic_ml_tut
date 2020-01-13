### 主题模型
假设手头上 N 篇文章这些文章，每一篇文章对应 k 主题，我们就可以用 k 维的主题概率分布来表示一篇文章。这样实质就是降维的过程。


想要了解隐含狄利克雷分布(Latent Dirichlet Allocation，以下简称LDA)模型我们需要对一下概念熟练掌握

#### 贝叶斯定理
贝叶斯模型遍布机器学习各个模型，还是先简单地回顾一下贝叶斯模型。贝叶斯模型是由 3 个部分组成
- 先验
- 后验
- 似然

这个很好理解，我们认识世界也是这个过程，根据自己从书本上学习到结合自己亲身经历就是我们对事物和世界的认知。贝叶斯就是将我们认知世界过程通过概率模型来描述出来。

还是投硬币示例来介绍我们认识投硬币这件事，我们在自己投硬币。根据经验认为投硬币正面和背面的概率各占一半都是 50% 概率，但是我们知己投硬币可能发现硬币正面朝上次数要大于背面朝上的次数。这样我们就需要调整参数

#### 二项分布与Beta

$$D(k|n,p) = \left( \begin{matrix}
    n\\
    k
\end{matrix} \right) p^k (1-p)^(n-k)$$
我们知道多次的 0 1 分布就是伯努利分布，这里 n 表示进行试验次数(也就是投掷硬币次数)而 k 表示我们期望事件出现次数(在投硬币试验中就是硬币正面朝上的次数) p 表示正面朝上的概率.


#### 共轭先验
共轭先验(conjugate priors)是一种概率密度，使得后验概率的密度函数和先验概率的密度函数有着相同的函数形式。

数据(似然)很好理解，但是对于先验分布，如何获得数据先验，通常我们会给出机会均等作为先验。因为希望这个先验分布和数据(似然)对应的二项分布集合后，得到的后验分布在后面还可以作为先验分布！是前面一次贝叶斯推荐的后验分布，又是后一次贝叶斯推荐的先验分布。也即是说，我们希望先验分布和后验分布的形式应该是一样的，这样的分布我们一般叫共轭分布。在我们的例子里，我们希望找到和二项分布共轭的分布。

在这里说题外话，也是个人学习机器学习的一种方法，如果一个知识点对于你可能理解上有点困难，这时候我们就需要先混个脸熟，这个和追女孩是一样的。首先要经常见面。

#### 伽马函数

我们对一些典型函数求积分我想大家可能非常熟悉
$$ \int x^t dx = \frac{x^{t+1}}{t+1}$$
$$ \int e^{-x} dx = e^{-x}  $$
$$ \int_0^{\infty} x^{t} e^{-x} dx$$
$$ \int_0^{\infty} t^{x} e^{-t} dt = x!$$

$$$$

$$\Gamma(x) = \int_0^{\infty} t^{x-1}e^{-t}dt = (x-1)!$$
伽马函数主要就是阶乘的连续性。
$$ x^2(1-x)^3 x \in [0,1] $$
这个函数知道在 0 到 1 之间，所以$x \ge 0$ x 等于 0 或者 x = 1 时候是 0 如图，那么如果

$$ \int_0^1 x^2(1-x)^3 dx $$ 就是这个曲线和坐标围成的面积那么将，

$$Beta(\alpha,\beta) = \frac{\Gamma(\alpha + \alpha)}{\Gamma(\alpha)\Gamma(\beta)} p^{(\alpha - 1)}(1 - p)^{(\beta - 1)}$$



仔细观察 Beta 分布和二项分布，可以发现两者的密度函数很相似，区别仅仅在前面的归一化的阶乘项。那么它如何做到先验分布和后验分布的形式一样呢？后验分布 $p(p|n,k,\alpha,\beta)$推导如下：


$$p(p|n,k,\alpha,\beta) $$
- p
- n
- k
- $\alpha$
- $\beta$
$$\begin{aligned}
    p(p|n,k,\alpha,\beta) \propto p(k|n,p)p(p|\alpha,\beta) & (1)\\
    = p(k|n,p)p(p|\alpha,\beta) & (2)\\
    = Binom(k|n,q)Beta(p|\alpha,\beta) & (3)\\
    = \left( \begin{matrix}
        n\\
        k
    \end{matrix} \right) p^k(1-p)^{n-k} \times \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} p^{\alpha - 1}(1-p)^{\beta - 1} & (4)\\
    \propto p^{k+\alpha-1}(1-p)^{n - k + \beta - 1}
\end{aligned}$$

$$p(p|n,k,\alpha,\beta) = \frac{\Gamma(\alpha + \beta + n)}{\Gamma(\alpha + k)\Gamma(\beta + n - k)} p^{k+\alpha - 1} (1-p)^{n - k + \beta -1}$$

可见后验分布的确是Beta分布，而且我们发现

$$Beta(p|\alpha,\beta) + BinomCount(k,n-k) = Beta(p|\alpha + k,\beta + n -k)$$

这个式子完全符合我们在之前投硬币例子里的情况，我们的认知会把数据里的正面、背面数分别加到我们的先验分布上，得到后验分布。

#### 多项分布与Dirichlet 分布
现在回到投硬币问题上，假设处理硬币有正面和背面情况以外还有硬正立这种情况
正面朝上$m_1$和背面朝上$m_2$ 和正立情况$(1 - m_2 - m_3)$



#### LDA 主题模型

LDA假设文档主题的先验分布是Dirichlet分布，即对于任一文档 [公式] ，其主题分布 [公式] 为



$$\begin{cases}
    f(x) = x^2 \\
    s.t. & x \ge b \Rightarrow x-b \ge 0
\end{cases}$$

$$L(x,\lambda) = x^2 - \lambda (x-b)$$

$$\begin{cases}
    \frac{\partial L}{\partial x} = 2x - \lambda = 0 \Rightarrow x = \frac{\lambda}{2}\\
    \frac{\partial L}{\partial \lambda} = b - x = 0 \Rightarrow x = b \Rightarrow \frac{\lambda}{2} = b \Rightarrow \lambda^{*} = 2b
\end{cases}$$

$$\lambda^{*} = \max(0,2b)$$
$$\begin{cases}
    x^{*} = 0 \\
    x^{*} = b
\end{cases}$$

$$\begin{aligned}
    \min_x \max_{\lambda} L(x,\lambda) = x^2 - \lambda(x-b)\\
    \max_{\lambda} \min_x L(x,\lambda) = x^2 - \lambda(x-b)\\
\end{aligned}$$

$$\begin{cases}
    \max = \frac{1}{2}|| w||^2 \\
    s.t. & y^{(i)}(w x^{(i)} - b  ) \ge 1\\
\end{cases}$$

$$\min_{w,b} = \frac{1}{2} ||w||^2 + \sum_{i=1}^N \alpha_i [1 - y^{(i)}(w x^{(i)} + b)]  $$


#### 特征向量和特征值

$$Ax = \lambda x \Rightarrow Ax = \lambda E x \Rightarrow (\lambda E - A)x = 0$$
$$A = \begin{bmatrix}
    4 & 2 & -5 \\
    6 & 4 & -9 \\
    5 & 3 & -7 \\
\end{bmatrix}$$

$$|\lambda E - A |= \begin{bmatrix}
    \lambda - 4 & -2 & 5 \\
    -6 & \lambda - 4 & 9 \\
    -5 & -3 & \lambda + 7 \\
\end{bmatrix}$$


$$|\lambda E - A | = (\lambda - 4)(\lambda - 4)(\lambda + 7) + (-2)\times 9 \times (-4) + (-6) \times (-3) \times 5 + 5 \times (\lambda - 4) \times (-5) + (-6) \times (-2) \times(\lambda + 7) + (-3) \times 9 \times (\lambda - 4) = 0$$

$$\lambda^2 (\lambda - 1) = 0$$

$$ \lambda_1 = 1 \, \lambda_2 = \lambda_3 = 0$$

#### $\lambda-1 = 1$

假设 A 是 n 阶方阵，若存在非 0  的数 $\lambda$ 和向量 x 满足
$$A x = \lambda x$$
- 这叫$\lambda$ 为 A 向量的特征值
-  x 叫做 A 的对应于 $\lambda$ 的特征向量

$$\lambda_1 = 1 \Rightarrow (E - A) x = 0$$
$$ E = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1 
\end{bmatrix}$$

$$ E - A = \begin{bmatrix}
    -3 & -2 & 5 \\
    -6 & -3 & 9 \\
    -5 & -3 & 8 
\end{bmatrix}$$

$$ E - A = \begin{bmatrix}
    -3 & -2 & 5 \\
    -6 & -3 & 9 \\
    -5 & -3 & 8 
\end{bmatrix}$$

$$ E - A = \begin{bmatrix}
    1 & 0 & -1 \\
    0 & 1 & -1 \\
    0 & 0 & 0 
\end{bmatrix}$$

$$ (E - A)x = \begin{bmatrix}
    1 & 0 & -1 \\
    0 & 1 & -1 \\
    0 & 0 & 0 
\end{bmatrix} \cdot \begin{bmatrix}
    x_1\\
    x_2\\
    x_3
\end{bmatrix}$$

$$\begin{cases}
    x_1 - x_3 = 0\\
    x_2 - x_3 = 0
\end{cases}$$

$$\epsilon_1 = \begin{bmatrix}
    1 \\
    1 \\
    1 
\end{bmatrix}$$

#### $\lambda_2 = \lambda_3 = 0$ 情况
$$(E - A)x = \begin{bmatrix}
    -2 & 0 & 1 \\
    0 & -2 & 3 \\
    0 & 0 & 0 
\end{bmatrix} \begin{bmatrix}
    x_1 \\
    x_2 \\
    x_3 
\end{bmatrix}$$

$$\begin{cases}
    2x_1 - x_3 = 0 \\
    2x_2 - 3x_3 = 0 
\end{cases}$$

$$\x_1 = 1$$

$$\epsilon_2 = \epsilon_3 = \begin{bmatrix}
    1 \\
    3 \\
    2
\end{bmatrix}$$

$$A = \begin{bmatrix}
    -1 & 1 & 0 \\
    -4 & 3 & 0 \\
    1 & 0 & 2
\end{bmatrix}$$

$$ |A - \lambda E| = \begin{bmatrix}
    -1 - \lambda & 1 & 0 \\
    -4 & 3 - \lambda & 0 \\
    1 & 0 & 2 - \lambda
\end{bmatrix}$$

以及每一个主题内容,如果一个幂函数和指数函数求积分
$$\int_0^{\infty} $$