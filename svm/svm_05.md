### SVM

优化问题

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

通常我们的问题是没有约束条件的优化，例如
$$\begin{aligned}
    min f(x) & x \in \mathbb{R} & or & x \in \mathbb{R}^n
\end{aligned}$$

我们可以引入等式 h(x) 约束条件来约束 f(x) 的优化问题，
$$\begin{cases}
    \min f(x) = 0 \\
    s.t. & h(x) = 0 
\end{cases}$$
当然我们约束条件可能不止一个约束条件可能是 i 个约束条件。
$$\begin{cases}
    \min f(x) = 0 \\
    s.t. & h_i(x) = 0 
\end{cases}$$

还可能有非等式的约束条件，例如我们添加小于等于0约束条件g(x)，可能你会问可以是大于等于 0  的约束条件不? 当然可以不过我们将大于等于 0 的约束条件转化为小于等于 0 的约束条件。
$$\begin{cases}
    \min f(x) = 0 \\
    s.t. & h_i(x) = 0 & g(x) \le 0
\end{cases}$$

$$\begin{cases}
    \min f(x) = 0 \\
    s.t. & h_i(x) = 0 & i = 1,2, \dots , n\\ 
    s.t. & g(x)_j \le 0 & j = 1,2, \dots m
\end{cases}$$
#### 无约束条件的优化问题
我们先将问题简化为无约束条件下求极值的优化问题，
$$ \min_x f(x) = x^2 (x  \in \mathbb{R})$$
$$ \begin{aligned}
    f(x) = x^2 \\
    \nabla_x f(x) = 0 \\
    \nabla_x f(x) = 0 \Rightarrow 2x = 0 \Rightarrow x = 0
\end{aligned}$$

上面推导我们在一个最简单方式求导过程，在导数为 0 位置也就是函数极值位置。多维空间,我们对每一个特征相对函数进行求导
$$\nabla_{x_j}f(x_j) = 0 $$

#### 等式约束条件优化问题
有关优化问题，我们先将约束条件从不等式简化为等式，将求极值问题转化为在等式约束条件下求极值问题。
$$\begin{cases}
    \min_x f(x) = x_1 + x_2 \\
    s.t. & h(x) = x_1^2 + x_2^2 -2 = 0
\end{cases}$$


就是在 $x_1^2 + x_2^2 -2 = 0$ 约束条件下求$f(x) = x_1 + x_2$ 函数的极值。

这张图是我精心绘制的，图中红色线表示 $x_1 + x_2 = 0$ 函数，我们这里以间隔 1 进行绘制，大家可以理解为函数f(x) 的等高线。蓝色圆表示等式约束条件。
其中 $x_1 + x_2 = 0$ 这条线穿过圆心的斜率为 -1 的直线。在任何一点都可以有一条平行直线，我们这里将这些间隔为 1 来选取直线理解为 f(x) 的等高线。

然后绘制出h(x)的图像，表示一个以$\sqrt{2}$ 半径的圆，这样我更直观地观察他们之间关系。这里约束条件就是要求我们$x_1$和$x_2$ 都要落在这个圆上，然后我们在这个圆上找到合适点来表示最小值。

什么样的点是极值点呢?假设点 $X_F$ 附近移动$\zeta_x$距离后满足下面条件我们就认为$X_F$ 是极值点，满足什么条件
$$\begin{cases}
    \delta_x: h(X_F + \mu \delta_x) = 0\\
    f(X_F + \mu \delta_x) < f(X_F)
\end{cases}$$

这里是 $\delta_x$ 无穷小的数，$\mu$ 表示每次移动$ \mu \delta_x$距离, $X_F$移动了$\mu \delta_x$距离后得到点依旧满足约束条件(落在圆上)，并且在函数值$f(X_F + \mu \zeta_x)$ 小于$f(X_F)$ 值。
有了移动距离，那么移动方向是要满足$ f(X_F + \mu \delta_x) < f(X_F)$ 因为我们希望每一次移动后都会变小，也就是沿梯度下降进行移动这样来找最小值。

$$\begin{aligned}
    f(x) = x_1 + x_2 \\
    \frac{\partial f(x_1,x_2)}{\partial x_1} = 1 \\
    \frac{\partial f(x_1,x_2)}{\partial x_2} = 1 \\
    \nabla_{x_1,x_2} f(x_1,x_2) = \begin{bmatrix}
        1 \\
        1
    \end{bmatrix}
\end{aligned}$$

因为f(x)中有两个变量，梯度也就是二维向量来表示分别在x1 和 x2 两个方向的梯度，我们知道梯度为零处函数出现极值，这里偏导向量$[1,1]^T$表梯度方向，因为损失函数通常是$\theta - \eta \frac{\partial J(\theta)}{\partial \theta}$ 所以我们更新是其负梯度。
图(svm_2)


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


那么在(1,-1)点的梯度为(2,-2),而在(-1,-1)点的梯度为(-2,-2),然后我们来看f(x)和h(x)两者之间的梯度关系。
$$\begin{cases}
    \delta_x \perp \nabla_x h(X_F) \\
    h(X_F + \delta_x) = 0 \\
    \delta_x \cdot (-\nabla_x f(X_f)) = 0

\end{cases}$$
通过上面不等式图我们
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


#### 1 情况 f(x)函数最小点落在限制条件g(x) 面积内
就可以按照非限制条件来正常选择
$$\nabla_x f(x) = 0$$
$$\Rightarrow \begin{matrix}
    2x_1 \\
    2x_2
\end{matrix} = 0$$

如果计算出全局最优解不满足限制条件，继续第二种情况
#### 2 情况
$$\begin{cases}
    f(x) = (x_1 - 1.1)^2 + (x_2 - 1.1)^2 \\
    g(x) \le 0
\end{cases}$$

### 不等式约束条件
$$L(x,\lambda) = f(x) + \lambda g(x)$$
$$\begin{cases}
    1. \nabla_x \perp (x^*,\lambda^*) \\
    2. \lambda^* \ge 0 \\
    3. \lambda^* g(x^*) = 0 \begin{cases}
        \lambda =0 \Rightarrow L(x,\lambda) = f(x) \\
        g(x) = 0 \Rightarrow \lambda > 0
    \end{cases} \\
    4. g(x^*) \le 0
\end{cases}$$

### 回归 SVM 问题
$$\begin{cases}
    \min_x f(x) \\
    s.t. & h_i(x) = 0 & i = 1,2, \dots , l \\
    s.t. & g_j(x) = 0 & j = 1,2, \dots , m 
\end{cases}$$

$$L(x,\mu,\lambda) = f(x) + \sum_{i=1}^l \mu_i h_i(x) + \sum_{j=1}^m \lambda_j g_j(x)$$

1. 条件
$$\begin{cases}
    \frac{\partial L}{\partial \vec{x}} = 0\\
    \frac{\partial L}{\partial \vec{\mu}} = 0\\
    \frac{\partial L}{\partial \vec{\lambda}} = 0
\end{cases}$$
2. 条件
$$\begin{cases}
    \lambda^*_j \ge 0 \\
    \mu_i \ge 0 \\
    \lambda_j^* g_j(x^*) = 0
\end{cases}$$

$$\begin{cases}
    \min_x x^2 \\
    s.t. & x \ge b (b-x \le 0)
\end{cases}$$

$$L(x,\lambda) = x^2 + \lambda(b - x)$$

我已经多次尝试去解释过 SVM ，每一次解释都是建立对 SVM 理解的一次更新基础上，今天再给我一次机会来说明清楚 SVM。我们在线性可分两类样本之间存在多条分隔线(超平面)。我们需要找到一最佳的超平面，那么怎么找到这个最佳超平面是问题关键。什么是最佳，也就是让两类点间真空地带尽量大，那么什么是真空地带。地带也就是需要两条线来进行划分，真空地带是由两条边界线划分出，这两条线满足两个条件一个条件就是满足平行于分隔超平面，并且他们至少通过一个样本点，分别位于超平面两侧，两个分隔线之间距离可能是 h ，我们可以通过对数据进行处理现象缩放。

我们用于数学语言描述一下问题就是我们有数据
$$D = ((x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\dots,(x^{(n)},y^{(n)})) x \in \mathbb{R}^n, y\in \{-1,1\}$$
$$\begin{cases}
    \theta^T x^{(i)} + b  > 0 & y^{(i)} =1 \\
    \theta^T x^{(i)} + b  < 0 & y^{(i)} =-1 
\end{cases}$$
我们如何判断一定给定样本点在模型中估计结果是否正确，这里用了一个小技巧 
$ y^{(i)} \theta^T x^{(i)} + b > 0 $ 说明我们估计值是正确的，这个不难理解当我们 $y^{(i)} = 1$ 如果函数 $\theta^T x^{(i)} + b > 0$ 说明我们分对了了。
$$y^{(i)} \theta^T x^{(i)} + b < 0$$
$$\begin{cases}
    wx + b = 0 \\
    wx + b = -1 \\
    wx + b = 1 
\end{cases}$$

线性方程
$$y=2x + 3$$

$$Wx + b = 0 $$
$$x_1 = x_2 + \lambda w$$


$$ x_2 -  x_1  = \lambda w$$
$$ x_2 = x_1 + \lambda w$$
$$ w(x_1 + \lambda w) + b = 1$$
$$ wx_1 + w^TW \lambda + b = 1$$
$$(wx_1 + b) +  w^TW \lambda  = 1$$
$$-1 +  w^TW \lambda  = 1$$
$$ w^TW \lambda  = 2$$
$$ \lambda = \frac{2}{||w||}$$

$$ \begin{cases}
    \frac{\partial L}{\partial \vec{x}} = 0 \\
\end{cases} $$

$$L(x,\mu,\lambda) = f(x) = \sum_{i=1}^l$$

$$\begin{cases}
\min_{x} x^2 \\
s.t. & x \ge b & (b - x \le 0)
\end{cases}$$

$$L(x, \lambda) = \underbrace{x^2}_{f(x)} + \lambda \underbrace{(b - x)}_{g(x)}$$

$$ \frac{\partial L}{\partial x} = 2x - \lambda = 0 \Rightarrow x^{*} = \frac{\lambda}{2}$$

$$\frac{\partial L}{\partial \lambda} = b - x = 0 \Rightarrow x = b \Rightarrow \frac{\lambda}{2} = b \Rightarrow \lambda^{*} = 2b $$

$$\lambda^{*} = \max(0,2b)$$
有两种情况如果全局最优解在约束条件内也就是$\lambda = 0$,最优解不在约束条件内外，那么$\lambda$ 就是一个不等于 0 的数时候就可以计算$\lambda = 2b$ 
$$x^{*} = \begin{cases}
    0 \\
    b
\end{cases}$$

比如说我们需要对于直线方程描述，高维空间有许多点，我们同时满足学多条件的
$$ \theta^T x + b  = 0 $$
$$ \theta^T x + b  < 0 $$
$$ \theta^T x + b  > 0 $$
分成完整连个部分，不过有很多种可能都可以将数据进行分离，不过我们需要找到一个完美的超平面将这些点进行分离。给定 $$ \theta^T x^{(i)} + b  > 0 y^{(i)} = 1 $$ 

我们可以构造

 $$ y^{(i)} (\theta^T x^{(i)} + b)  > 0 $$ 
 $$ y^{(i)} (\theta^T x^{(i)} + b)  < 0 $$
 我们给定线性
 $$y = 2x + 3$$
我们这里用向量形式来表示方程
 $$(2,1) \left( \begin{matrix}
     x \\
     y
 \end{matrix} \right) + 3  = 0$$ 

 这里是$\theta^T$ 表示转置也就是直线垂直的一条向量。代表线性函数的法线方向也就是和这条直线垂直的方向。







### 线性模型
$$\begin{aligned}
    w \cdot x - b = 0 \\
    w \cdot x_i - b \ge 0 & if \, y_i =1 \\
    w \cdot x_i - b \le 0 & if \, y_i = -1\\
    y_i (w \cdot x_i - b) \ge 0 \\
\end{aligned}$$

### 损失函数（Hinge Loss)
$$ l = \max(0,1-y_i(w \cdot x_i - b)) $$

$$l = \begin{cases}
    0 & if \, y\cdot f(x) \ge 1 \\
    1 - y \cdot f(x) & otherwise
\end{cases}$$

### 添加限制条件
$$ J = \lambda ||w||^2 + \frac{1}{n} \sum_{i=1}^n \max(0,1-y_i(w \cdot x_i -b ))$$

#### 第一种情况
$$if y_i \cdot f(x) \ge 1$$
$$ J_i = \lambda ||w||^2$$
#### 第二种情况
$$ J = \lambda ||w||^2 + 1 - y_i(w \cdot x_i - b)$$

### 梯度
#### 第一种情况
$$ if \, y_i f(x) \ge 1 $$

$$\begin{cases}
    \frac{d J_i}{d w_k} = 2 \lambda w_k \\
    \frac{d J_i}{ d b} = 0
\end{cases}$$
#### 第二种情况
$$\begin{cases}
    \frac{d J_i}{d w_k} = 2 \lambda w_k - y_i \cdot x_i\\
    \frac{d J_i}{ d b} = y_i
\end{cases}$$


### 更新规则
$$ w = w - \alpha $$