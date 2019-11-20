### 二项逻辑回归
- 最终输出0到1之间的值，可以解决类似"成功或失败","是否存在"或是"通过或拒绝"等类似的问题
- 逻辑回归是一个把线性回归问题模型映射为概率模型，即把实数空间输出$[- \infty,+\infty]$ 映射到(0,1),从而获得概率。
- 对于一元线性方程模型:$\hat{y} = wx + b$ 通过 b 值变化可以上下或左右移动直线，w 改变直线斜率
- 定义二元线性模型:$\hat{y} = \theta_1 x_1 + \theta_2 x_2 + b$ 我中$\hat{y}$取值范围在$[- \infty,+\infty]$


$$\theta_1 x_1 + \theta_2 x_2 + b$$
我们需要将取值在 $(-\infty,+\infty)$ 映射到$[0,1)$区间
$$\frac{正例样本数量}{负例样本数量}$$ 

对于$\frac{p}{1 - p}$ 取log那么就得到 $\log \frac{p}{1 - p} $ 现在$f(q) = \log \frac{p}{1 - p} $ 在
$$ \log \frac{p}{1 - p} = z $$
$$ \theta_1 x_1 + \theta_2 x_2 + b = z $$
$$ e^z = \frac{p}{1-p}$$
$$ (1-p) e^z = p$$
$$  e^z = p + p e^z$$
$$  e^z = p + p e^z$$
$$  p = \frac{e^z}{1+e^z}$$
$$  p = \frac{1}{\frac{1}{e^z}+1}$$
$$  p = \frac{1}{e^{-z}+1}$$

同时满足这两个限制条件的最优化问题称为**凸优化问题**，这类问题有一个非常好性质，那就是局部最优解一定是全局最优解。接下来我们先介绍**凸集**和**凸函数**的概念。
$$\theta x + (1 - \theta) y \in \mathbb{C}$$
![](https://github.com/zideajang/basic_ml_tut/blob/master/wx/logistic_regression/screenshots/context_1.JPG)
则称该集合称为凸集。如果把这个集合画出来，其边界是凸的，没有凹进去的地方。直观来看，把该集合中的任意两点用直线连起来，直线上的点都属于该集合。相应的，点：

$$ f(\theta x + (1 - \theta y)) \le f(\theta x) + f((1 - \theta) y)$$


$$ f(x,y,z) = 2x^2 - xy + y^2 - 3z^2$$

$$ \left[ 
 \begin{matrix}
    \frac{\partial^2f}{\partial x^2} && \frac{\partial^2f}{\partial x \partial y} && \frac{\partial^2f}{\partial x \partial z} \\
    \frac{\partial^2f}{\partial y \partial x } && \frac{\partial^2f}{\partial y^2 } && \frac{\partial^2f}{\partial y \partial z} \\
    \frac{\partial^2f}{\partial z \partial x} && \frac{\partial^2f}{\partial z \partial y} && \frac{\partial^2f}{\partial z^2}
\end{matrix} 
\right]
$$

1. Hessian 矩阵正定，函数在该点有极小值
2. Hessian 矩阵负定，函数在该点有极大值
3. Hessian 矩阵不定，还需要看更高阶的导数

#### 局部最优解与全局最优解
对于一个可行点 x，如果在其邻域内没有其他点的函数值比该点小，则称该点为局部最优，下面给出这个概念的严格定义：对于一个可行点，如果存在一个大于 0 的实数 $\delta$，对于所有满足：

$$||x - z ||_2 \le \delta $$
$$f(z) < f(x)$$
$$ z= \theta x + (1 - \theta) y $$
$$ \theta = \frac{\delta}{2 ||x-y||_2} $$