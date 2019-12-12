### SVM




$$\begin{cases}
    f(x) = x_1^2 + x_2^2 \\
    s.t. & g(x) = x_1^2 + x_2^2  \ge 0
\end{cases}$$

$$ f(x) = x_1^2 + x_2^2$$
### 我们将问题扩展到不等式
不等式其实就是一个区域，我们等式就是将一个约束条件等式曲线投影到曲面上，然后我们曲面极值点不但要满足是曲面极值点还要满足其落在曲线上。
#### 第一种情况
假设我们函数全局最最小值落在
#### 第二种情况

首先我们了解一下什么是混合模型 a 和 b
$$f(x) = \theta$$

$$\mu_j = $$

今天我们来学习 EM 我么看到变化背后隐含变量。

![task.jpg](https://upload-images.jianshu.io/upload_images/8207483-accde6b60e6e93b9.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

鸣人(B)和佐助(C)完成任务，具体是由谁完成任务是卡卡西(A)进行分别的，将任务分配给鸣人的概率为(pi)如果卡卡西(A)指派鸣人(B)来完成任务,他完成任务概率为 p ，如果是佐助(B)完成任务概率为 q。那么我们看到结果受到隐含变量 A 的影响。

$$\begin{aligned}
    P(Y|\theta) = \prod_i^N P(y^{(i)}|\theta) \\
    = \prod_i^N \pi p^{y^{(i)}}(1-p)^{1-y^{(i)}} + (1 - \pi) q^{y^{(i)}}(1-q)^{1-y^{(i)}} \\
    = \sum_{i=1}^N \log P(y^{(i)}|\theta)
\end{aligned}$$

上面公式应该不难理解，假设概率是独立的，可以通过假设 B 的概率为 $\pi$ 那么对于 $y^{(i)}$ ，如果  $y^{(i)}$ 等于 1 那么概率就是 B 完成任务概率就是 q 如果$y^{(i)}$ 等于 0 概率也就是 1 - q。

我们需要求出$\pi,p,q$ 概率，完成任务用 1 表示而 0 表示失败。如果我们假设知道每一次任务是由 B 和 C 谁来完成任务的话，问题就简单了。如下表

| B  | B  | B |C  | C  | C | B  | C  | B |
|---|---|---|---|---|---|---|---|---|---|
|  1 | 1  | 0  | 1  | 0  |  1 | 1  | 0  | 0  | 1  |


有了上面已知条件，这件事给定参数$\pi,p,q$ 参数

$$\begin{cases}
    p = \frac{count(1)}{naruto} \\
    q = \frac{count(1)}{sasuke}
\end{cases}$$
上面分别是，也就是找到任务中多少个是鸣人(naruto)来完成，1 表示鸣人成功完成任务的个数，也就是 p
$$\pi = \frac{naruto}{naruto + sasuke} = \frac{naruto}{N}$$

下面是 pi 的估计 N 表示任务数量，其中由鸣人(naruto)完成任务数占总任务数的比就是 pi。

所以我们需要对每一次结果来推测这个任务是 B 还是 C 完成，我们可以近似估计.

| B  | B  | B |C  | C  | C | B  | C  | B |
|---|---|---|---|---|---|---|---|---|---|
|  1 | 1  | 0  | 1  | 0  |  1 | 1  | 0  | 0  | 1  |
|  0.5 | 0.3  | 0.2  | 0.3  | 0.2  |  0.3 | 0.3  | 0.2  | 0.2  | 0.3  |

最后一行使我们在不知道每一次任务具体事由随来做的时候，对该次任务是由鸣人完成的概率推测。

$$\mu_j =  \frac{\pi p^{y^{(i)}}(1-p)^{1-y^{(i)}} }{ \pi p^{y^{(i)}}(1-p)^{1-y^{(i)}} + (1 - \pi) q^{y^{(i)}}(1-q)^{1-y^{(i)}}}$$

表示进行 i 任务 然后根据$\mu$ 来估计 $\pi p q$

$$\pi = \sum_{\mu_j}$$

可能很多人因为其复杂算法，而放弃学习 SVM。我想说这不一定是你的错，可能是你没有遇到好的老师。我也是遇到一位好老师才把 SVM 问题搞了七七八八，希望我的语言能够对于您了解 SVM 问题有所帮助。

在开始之前我们大概整理一下思路，首先我们是求一个函数的极值问题，也就是优化问题。然后我们将问题增加难度，我们是在一个等式约束条件下求函数极值问题，这也就是我们熟悉拉格朗日乘子法。最后我们我们等式约束扩展不等式约束来再次提高优化问题的难度。

代码我已经分享到 github 虽然不算专业，

### 优化问题
我们先从优化问题讲起，只有了解不等式约束条件的优化问题，也就是拉格朗日算法我们才能真正理解 SVM 算法。

$$\begin{cases}
    \min f(x) = 0 \\
    s.t. & h(x) = 0 \\
    x \in \mathbb{R}
\end{cases}$$
我们首先要学会用数据模型描述问题，这其实就是建模的过程。这里我们是求 f(x) 曲线最小问题，这个优化求最小问题是建立在 h(x) = 0 约束条件下。

如果将问题扩展到多维空间中表示如下，然后我们知道约束条件可以是等式也可以是不等式。
$$\begin{cases}
    \min f(x) = 0 \\
    s.t. & h_i(x) = 0  & i = 1,2, \dots, l\\
    & g_j(x) \le 0 & j = 1,2, \dots, m \\
    x \in \mathbb{R}^n
\end{cases}$$

我们先将问题简化为无约束条件下求极值的优化问题，
$$ \min_x f(x) = x^2 (x  \in \mathbb{R})$$
$$ \begin{aligned}
    f(x) = x^2 \\
    \nabla_x f(x) = 0 \\
    \nabla_x f(x) = 0 \Rightarrow 2x = 0 \Rightarrow x = 0
\end{aligned}$$

上面推导我们在一个最简单方式求导过程，在导数为 0 位置也就是函数极值位置。

#### 等式约束条件优化问题
有关优化问题，我们先将约束条件从不等式简化为等式，将求极值问题转化为在等式约束条件下求极值问题。
$$\begin{cases}
    \min_x f(x) = x_1 + x_2 \\
    s.t. & x_1^2 + x_2^2 -2 = 0
\end{cases}$$

就是在 $x_1^2 + x_2^2 -2 = 0$ 约束条件下求$f(x) = x_1 + x_2$ 函数的极值。

这张图是我精心绘制的，图中红色线表示 $x_1 + x_2 = 0$ 函数，我们这里以间隔 1 进行绘制，大家可以理解为函数f(x) 的等高线。蓝色圆表示等式约束条件。

什么样的点是极值点呢?假设点 $X_F$ 附近移动$\zeta_x$距离后满足下面条件我们就认为$X_F$ 是极值点，满足什么条件
$$\begin{cases}
    \zeta_x: h(X_F + \mu \zeta_x) = 0\\
    f(X_F + \mu \zeta_x) < f(X_F)
\end{cases}$$
通过也就是移动$\zeta_x$距离后得到点依旧满足约束条件，并且在函数值$f(X_F + \mu \zeta_x)$ 小于$f(X_F)$ 值。
$$\begin{aligned}
    f(x) = x_1 + x_2 \\
    \frac{\partial f(x_1,x_2)}{\partial x_1} = 1 \\
    \frac{\partial f(x_1,x_2)}{\partial x_2} = 1 \\
    \nabla_{x_1,x_2} f(x_1,x_2) = \begin{bmatrix}
        1 \\
        1
    \end{bmatrix}
\end{aligned}$$

我们通过计算向量来看量，我们知道梯度为零处函数出现极值，这里偏导向量$[1,1]^T$表梯度方向，因为损失函数通常是$\theta - \eta \frac{\partial J(\theta)}{\partial \theta}$

$$\begin{cases}
    \vec{a} \cdot \vec{b} > 0 \\
    \vec{a} \cdot \vec{b} = 0 \\
    \vec{a} \cdot \vec{b} < 0 
\end{cases}$$

$$\begin{cases}
    \frac{\partial h(x_1,x_2)}{\partial x_1} = 2 x_1 \\
    \frac{\partial h(x_1,x_2)}{\partial x_2} = 2 x_2 \\
\end{cases} \Rightarrow \begin{bmatrix}
    2 x_1 \\
    2 x_2 
\end{bmatrix}$$

$$-\nabla_x f(X_F) = \mu \nabla_x h(X_F)$$

$$\begin{cases}
    min_x f(x) \\
    s.t. & h(x) = 0
\end{cases} \Rightarrow L(x,\mu) = f(x) + \mu h(x)$$
这里$\mu$ 就是拉格朗日乘子，我们

$$\frac{\partial L}{\partial x} = \nabla_x f(x) + \mu \nabla_x h(x)$$

### 不等式
$$\begin{cases}
    min_x f(x) \\
    s.t. & g(x) \le 0
\end{cases}$$

$$\begin{cases}
    f(x) = x_1^2 + x_2^2 \\
    g(x) = x_1^2 + x_2^2 - 1= 0
\end{cases}$$

从上面例子来看，很清楚我们限制条件就是 f(x) 一部分并且全局最优解落在这个限制区域内。