可以用拉格朗日来解决约束条件问题，我们这里有多少个样本N就有多少

$$ G(x, r,\lambda) = f(x) + \sum_{i=1}^n \lambda_i f_i(x) + \sum_{j=1}^m r_j h_j(x)$$
$$ \lambda_i =  f_i(\lambda_i)$$
我们想象一下，可以将转换为关于$\lambda_i$的方程，其他都可以暂时看成参数，那么就是一个关于$\lambda_i$的线性方程，如下图，对函数求最小值就形成了一个凸函数，在凸函数上一定有一个最大值，因为
凹函数优化，有关凹函数优化问题，
$$ \lambda_1, \lambda_2$$
我们想象一下，分别在$x_0,x_1, \cdots x_n$ 在每一个点位置对应最小值如图
$$ min G(x, \mu \lambda) = min f(x)$$


### 几何意义
支持向量机评估标准，分割面向一侧分类点靠拢，知道通过一些点，这个线需要平移。那么这里平面为支撑平面，在2维为直线而在3为，那么如果两个平面达到最大值我们就说这个里，这隔离地带越宽也就是越好，这就是支持向量机几何意义。我们分隔平面位于两个支撑平面之间。
支持支持向量机，一般都都会经过一个支持一点，如果经过两个就是一种巧合。

我们用向量来表达直线
$$\vec{v} \cdot \vec{x} + b$$
有关内积介绍，几何意义为两个向量$\vec{a}$ 也就是$||a_1|||a_2||\cos \theta$ 也就是几何空间的知识。

### 推导支撑平面
w 平面的法向量也就是与分割平面垂直的向量，
$$\begin{aligned}
    w_0x_1 + b_1 = 0 \\
    w_0x_2 - b_2 = 0 
\end{aligned}$$

因为这两条直线具有相同斜率，
$$ \begin{aligned}
    wx+b = -1 \\
    wx+b = 1
\end{aligned} $$

$$ \begin{aligned}
    wx_1+b = -1 \\
    wx_2+b = 1
\end{aligned} $$

$$
    \begin{aligned}
        w \cdot (x_1 - x_2) = 2 \\
        ||w|| \cdot ||x_1 - x_2|| \cos \theta = 2 \\
        ||w|| \times d = 2 \\
        d = \frac{2}{||w||}
    \end{aligned}
$$

### 转化为凸优化问题
$$
\begin{cases}
    w \cdot x_i + b \ge 1 & y_i = 1 \\
    w \cdot x_i + b \le 1 & y_i = -1 
\end{cases}
$$

$$ d = \frac{2}{||w||}$$

如果我们要数学上表示出这些表达式就有上面

$$\begin{aligned}
    \min_w \frac{||w||^2}{2} \\
    s.t. & y_i(w \cdot x_i + b) \ge 1, & i =1,2, \cdots , N 
\end{aligned}$$

### 凸优化问题
- 可以寻求凸优化
- 使用拉格朗日乘子算法
#### 什么是凸函数
$$f(x \theta + (1 - \theta)) y $$
$$y_i(wx_i + b) \ge 1$$
$$\min_w \frac{||w||^2}{w}$$

#### 拉格朗日乘子法
原理
$g(x,y)=c$
虚线可以看成等高线 ，等高线，极限值，一直扩展到就是我们需要，在几何上意义上相切的。红色向量刚好垂直向量，必须共线，



$$\begin{cases}
    \min f(x,y) \\
    g(x,y) = c
\end{cases}$$

$$L(x,y) = f(x,y) + \lambda (g(x,y) - c)$$
$$\nabla L = \nabla (f + \lambda (g-c))$$
$$\begin{aligned}
    \nabla L = \nabla (f + \lambda (g-c)) \\
    = \nabla f + \lambda \nabla g = 0 \\
    \nabla f = - \lambda \nabla g
\end{aligned}$$
KKT

$$L(w,b,a) = \frac{1}{2} ||w||^2 - \sum_{i=1}^n \alpha_i(y_i(w^Tx_i + b) -1)$$
KKT 乘子法
$$\begin{cases}
    \frac{\partial L}{\partial w} \Rightarrow \sum_{i=1}^n \alpha_i y_i x_i \\
    \frac{\partial L}{\partial b} \Rightarrow \sum_{i=1}^n \alpha_i y_i = 0
\end{cases}$$
### 梯度
$$\nabla L(x,y) = (\frac{\partial L}{\partial x},\frac{\partial L}{\partial y})$$