### 数学

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

### 泰勒公式和拉格朗日公式(Maclaurin 公式)


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