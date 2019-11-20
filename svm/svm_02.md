可以用拉格朗日来解决约束条件问题，我们这里有多少个样本N就有多少

$$ G(x, r,\lambda) = f(x) + \sum_{i=1}^n \lambda_i f_i(x) + \sum_{j=1}^m r_j h_j(x)$$
$$ \lambda_i =  f_i(\lambda_i)$$
我们想象一下，可以将转换为关于$\lambda_i$的方程，其他都可以暂时看成参数，那么就是一个关于$\lambda_i$的线性方程，如下图，对函数求最小值就形成了一个凸函数，在凸函数上一定有一个最大值，因为
凹函数优化，有关凹函数优化问题，
$$ \lambda_1, \lambda_2$$
我们想象一下，分别在$x_0,x_1, \cdots x_n$ 在每一个点位置对应最小值如图
$$ min G(x, \mu \lambda) = min f(x)$$