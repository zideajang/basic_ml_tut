### 数学
看过了很多书籍和视频，虽然对机器学习有了一点了解，还不敢说一定。但是感觉还是有必要捡以前的知识，我们回头看一看用到了那些数学和概率知识。

学点数学对于程序员是没有坏处的，让我们思维更加怎么，通过数学模型将一些业务抽象化，可以避免一些开发过程中问题。

概率也不是没有帮助，让我们考虑问题更全面，所有学习概率和统计学即使没有机会从事机器学习整个行业也会下意识地提供我们开发代码能力。


首先我们看看 e 这个数字

$$ \hat{y} = \theta_1 x_1 + \theta_2 x_2 + \dots \theta_i x_i $$

$$ L(\theta) = \sum_{i=1}^m (f(\theta_i,\theta_i) - y_i) $$

$$ s=\frac{1}{0!} + \frac{1}{1!} + \cdots + \frac{1}{n!} + \cdots $$

$$ f(x) = \log_a x $$

$$ \frac{f(x - \Delta x) - f(x)}{\Delta x} $$
$$ = \frac{\log_{a}(x - \Delta x) - \log_{a}(x)}{\Delta x} $$
$$ = \frac{\log_a \left( \frac{x + \Delta x}{ x} \right) }{\Delta x}$$
$$ = \log_a \left( \frac{x+ \Delta x}{x} \right)^{\frac{1}{\Delta x}}$$
$$ \rightarrow \log_a(1 + \Delta x)^{\frac{1}{\Delta x}} == 1 \Rightarrow \lim_{\Delta \rightarrow 0} (1 + \Delta x)^{\frac{1}{\Delta x}} = \alpha$$
$$ \lim_{n \rightarrow \infty} ( 1 + \frac{1}{n})^n$$

$$ x_n = \left( 1+ \frac{1}{n}  \right)^n $$
$$ = 1 + C_n^1 \frac{1}{n} + C_n^2 \frac{1}{n^2} + C_n^3 \frac{1}{n^3} + \cdots + C_n^n \frac{n}{n^n}$$

$$ 1 + n \cdot \frac{1}{n} + \frac{n(n-1)}{2!} \cdot \frac{1}{n^2} + \cdots $$

$$ 1 + 1 + \frac{1}{2!} \cdot \left( 1 - \frac{1}{n} \right) + \frac{1}{3!} \cdot \left( 1 - \frac{1}{n} \right)\left( 1 - \frac{2}{n} \right)  \cdots + \frac{1}{n!} \cdot \left( \frac{}{} \right)$$
$$$$
### 导数



#### 常用导数
$$ (C) \prime = 0$$
$$ (x^{\mu}) \prime = \mu x^{\mu - 1} $$

$$ (\sin x) \prime = \cos x $$
$$ (\cos x) \prime = - \sin x $$

#### 导数
#### 常用函数的导数
导数会在损失函数部分会用到导数，
- $f(x) = x^x, x>0$
$$ t = x^x $$
$$ \ln t = x \ln x $$
$$ t= e^{-\frac{1}{e}} $$
- $N \rightarrow \infty \Rightarrow \ln N! \rightarrow N(\ln N - 1)$
$$ln N! = \sum_{i=1}^N \ln i \approx \int_1^N \ln xdx $$

#### 梯度计算
$$ \frac{\partial f(x,y)}{\partial x} \times \cos + \frac{\partial f(x,y)}{ \partial y} \times \sin x$$

我们都法线方向

$$ y = x^2 $$
$$ y = 2x $$
$$ y = 2 $$

如果一个函数进行二阶求导后是大于0 说明该函数就是凸函数，凸函数就是可以用到梯度优化算法，这个想必大家都清除。那么对于二元的函数我们又应该如何判断是否为凸函数，下面我们来具体在看一看如何判断二元二阶函数是否是**凸函数**。

$$ 2x^2 + 3y^2 + xy $$
先对 x 求偏导数
$$ \frac{\partial f(x,y)}{ \partial x} = 4x + y $$
然后对 y 进行求偏导数
$$ \frac{\partial f(x,y)}{\partial y} = 6y + x $$

然后进行二阶导数
$$ \frac{\partial(x,y)}{\partial x \partial x} = 4$$
$$ \frac{\partial(x,y)}{\partial y \partial y} = 6$$



### 泰勒公式和拉格朗日公式(Maclaurin 公式)
$$ e^x =  1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots + \frac{x^k}{k!} + R_k$$

$$ 1 =  1 \cdot e^{-x}  + x \cdot e^{-x} + \frac{x^2}{2!}\cdot e^{-x}  + \frac{x^3}{3!} \cdot e^{-x} + \cdots + \frac{x^k}{k!} \cdot e^{-x}  + R_k \cdot e^{-x} $$

$$ \frac{x^k}{k!} \cdot e^{-x} \rightarrow  \frac{\lambda^k}{k!} \cdot e^{-\lambda} $$

如果将 X 看做一个常数项，而将 k 看做我们研究对象那么，而且保证加和为 
#### 泊松分布
这样东西也是就大家熟悉泊松分布
假设$X \approx \pi(\lambda)$ 且分布律为泊松分布

$$ P{X = k} = \frac{\lambda^k}{k!} e^{-\lambda} , k = 0,1,2, \cdots , \lambda > 0$$
则有
$$ E(X) = \sum_{k=0}^{\infty} k \cdot \frac{\lambda^k}{k!} e^{-\lambda} = e^{\lambda} \sum_{k=1}^{\infty} \frac{\lambda^{k-1}}{(k-1)!} \cdot \lambda = \lambda e^{} $$
#### 均匀分布
概率密度函数
$$ f_x(x) = \begin{cases}
    \frac{1}{b-a} & a \le x \le b \\
    0 & otherwise
\end{cases} $$

CDF
$$F_x(x) = \int_{-\infty}^x f_x(u)du = \begin{cases}
    0 & x \le a \\
    \frac{x - a}{b - a} & a < x \le b \\
    1 & x > b 
\end{cases}$$
#### 指数分布

$$ f(x) = \begin{cases}
    \frac{1}{\theta} e^{-\frac{x}{\theta}} & x > 0 \\
    0 & x \le 0
\end{cases} $$
##### 指数分布的无记忆性

#### 正态分布


## 概率
## 分布

### 两点分布(伯努利分布)
设随机变量 X 服从
| X  |  1 | 0|
|---|---|---|
|  P |  p | 1-p|
期望
$$ E(X) = 1 \cdot p + 0 \cdot q = p $$
方差
$$ D(X) = E(X^2) - [E(X)]^2$$
$$ = 1^2 \cdot p + 0^2 \cdot (1 - p) - p^2 = p(1-p) = pq$$
### 二项分布
可以看做若干个两点分布的取和，设随机变量 X 服从参数为 n,p 二项分布, 假设 $X_i$ 为第 i 次试验中事件 A 发生的次数，$i=1,2,\cdots , n$
$$ X = \sum_{i=1}^n X_i$$
显然，$X_i$相互独立均服从参数为 p 的 0-1 分布所以
$$ E(X) = \sum_{i=1}^n E(X_i) = np$$
$$ D(X) = \sum_{i=1}^n D(X_i)= np(1-p)$$
在做分类问题，将样本为样本有

我们假设有样本,有些样本是正例有些样本是负例。取 1 概率 p 取 0 概率为 0 
$$ X = \begin{cases}
    p & 1 \\
    1 - p & 0
\end{cases} $$

$$ X \{ (x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m) \}$$
这里有 m 个样本，其中$y \in \{0,1\}$ 也就是样本类别为，其实这就是 m 次试样的两点分布。拿出其中一个 $x_i$ 

$$ = p^1 \cdot (1-p)^{(1 -0)} \tag{1}$$
这是两个分布统一
$$ = p^{y_i} \cdot (1-p)^{(1 -y_i)} \tag{2}$$
如果假设$x_i$ 样本具有 n 维 $(x_{i1},x_{i2},\dots,x_{in})$
| $x^{i}$  | $\theta_1$  |$\theta_1$  | $\cdots$ |$\theta_n$  |
|---|---|---|---|---|
| $x^{(1)}$  | $x_1^1$  |  $x_2^1$ | $\cdots$ | $x_n^1$|
| $x^{(2)}$  | $x_1^2$  |  $x_2^2$ | $\cdots$ | $x_n^2$|
| $x^{(n)}$  | $x_1^n$  |  $x_2^n$ | $\cdots$ | $x_n^n$|


$$\vec{\theta} \cdot \vec{x_i}$$
然后将上面式子导入$\frac{1}{1+ e^{-x}}$ 这个公式大家再熟悉不过我们在做逻辑回归时候看到sigmoid 公式。
$$p = \frac{1}{1+ e^{-(\vec{\theta} \cdot \vec{x_i})}} \tag{3}$$
把这个概率带入上面(2)式里我们就得到关于$\theta$的目标函数，当这个目标函数最大时候哪个$\theta$就是我们要找参数，也就是给定目标函数一种手段。

例如电影会有许多主题例如奇幻、动作、奇幻

### 协方差的上限
$$ Var(X) =  $$

### 中心极限定理
假设随机变量$X_1,X_2, \dots , X_n , \dots $ 相互独立，服从同一分布，


$$ f(x) = \frac{f(x_0)}{0!} + \frac{f \prime (x_0)}{1!}(x - x_0) +  \frac{f \prime \prime (x_0)}{2!}(x - x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x - x_0)^n  + R_n(x)$$

### 流程
- 数据收集
- 数据清洗
- 特征工程
- 数据建模
- 数据模型应用



### 概率论与贝叶斯先验
先验概率就是根据我们已知的知识和经验给出概率
**本福特定律**又称第一数字定律，是在实际生活得出的一组数据中，

### 商品推荐
| A  | B  |
|---|---|
|  0.8 | 0.2  |

计算B的分数大于A的分数的概率

A=B
$$ \begin{cases}
    S_{蓝色} = 0.02 \\
    S_{矩形} = 0.16 
\end{cases} $$
$$p = \frac{0.02}{0.16} = 0.125$$

### 两点分布(0-1 分布)
| X  |  1 | 0  |
|---|---|---|
| p  | p  | 1 - p  |

$$ E(x) = 1 \cdot p + 0 \cdot q = p$$
$$ D(X) = E(X^2) - [E(X)]^2 = 1^2 \cdot p + 0^2 \cdot (1-p) - p^2 = pq$$

### 二项分布
- 期望 $E(x) = np$
- 方差 $D(x) = npq = np(1-q)$
#### 期望
$$ X \{ x_1,x_2, \dots ,x_i \} $$
$$ P \{ p_1, p_2, \dots , p_i \}$$
$$ E(x) = x_1p_1, x_2p_2 \dots x_np_n$$
加权平均值
#### 方差
$$ D(x) = (x_1 - E(x))^2  p_1  + (x_2 - E(x))^2  p_2  + \dots (x_n - E(x))^2  p_n $$
加权方差
$$ D(x) = E(x^2) - [E(x)]^2 $$
平方期望减去期望平方