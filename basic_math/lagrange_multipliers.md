### 拉格朗日乘子


拉格朗日乘子法是一种寻找多元函数在一组约束下的极值方法。

![larange_01.jpg](https://upload-images.jianshu.io/upload_images/8207483-8066bd97dee0bbbb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图中
与椭圆体相交平面上直线$g(x,y)$如果高度上没有限制那么$g(x,y)$就形成一个面，这个面与椭圆体相交可以表示为$z=f(x,y)$,我们就可以在这个曲线找到最小值。然后我们可以将这等高线投影到二维平面上来简化问题

![larange_multiplier_3.jpeg](https://upload-images.jianshu.io/upload_images/8207483-0ae165b63afe7269.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在上图中，我们可以推断出其实最小(或最大值)就位于限制条件g(x,y)和方程f(x,y)等号线相切的位置。而且有共同切线的斜率，那么他们法线方向是**成比例**的。这个比例系数就是拉格朗日乘子
$$ \nabla f = \lambda \nabla g $$

我们现在来简单推导一下，这里将 y 表示为对于 x 的函数，那么就有 y(x),然后分别带入下面两个方程就得到。
$$ g(x,y(x)) = 0$$
$$ f(x,y(x))$$

下面我么这个两个方程都对x 进行偏微分，通过链式法则我们就得到下面式子

$$\frac{\partial f}{\partial x} = f_x + f_y \frac{\partial y}{\partial x}$$

$$\frac{\partial g}{\partial x} = g_x + g_y \frac{\partial y}{\partial x}$$

因为我们知道他们斜率是成比例的，所有就可以得到这样结论，这就是拉格朗日乘子法，其中$\lambda$就是乘子

$$ \exists \lambda \begin{cases}
    f_x = \lambda g_x  \\
    f_y = \lambda g_y \\
    g(x,y) = 0
\end{cases} $$

我们就可以利用这个三个条件来求在有限制条件下方程极值问题

### 例题
假设$f(x,y) = 3x + 4y$,在$x^2 + y^2 - 1 = 0$ 的条件限制下有极值。
利用上面知识来求极值
$$ \begin{cases}
     f_x = \lambda g_x & 3 = \lambda 2x  \\
    f_y = \lambda g_y & 4 = \lambda 2y \\
    g(x,y) = 0 & x^2 + y^2 = 1
\end{cases}$$

$$ x = \frac{3}{2 \lambda} , y = \frac{4}{2 \lambda}$$
然后他们带入到$x^2 + y^2 - 1 = 0$ 得到

$$ \frac{9}{4 \lambda^2} + \frac{16}{4 \lambda^2} = 1$$
$$ \lambda = \pm \frac{2}{5}$$

$$\begin{cases}
    \lambda = \frac{2}{5} & x = \frac{5}{3} y =\frac{5}{4} \Rightarrow f(\frac{5}{3},\frac{5}{4}) = 5 \\
    \lambda = \frac{2}{5} & x = -\frac{5}{3} y =- \frac{5}{4} \Rightarrow f(-\frac{5}{3},-\frac{5}{4}) = -5
\end{cases}$$

那么结果就是最小值和最大值分别是 5 和 -5

### 有约束凸优化的优化问题求解
可行域上的最优解一定会碰到一个边界，即至少有一个c(x) = 0
在约束条件下，可行域围成的区间为抛物线以上，直线以下的那部分区域。最优解必须在可行域中，最优解为 A 点，最优解会碰到
### 拉格朗日

### 拉格朗日对偶函数
$$
\begin{cases}
    min & f_0(x) \\
    s.t. & f_i(x) \le 0 & i =1,\cdots ,m \\
    s.t. & h_j(x) = 0 &  j = 1, \cdots , n
\end{cases}
$$

$$\nabla f_0(x^{*}) = \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + \sum_{i=1}^n \nu_j^* \nabla h_j(x^*) $$