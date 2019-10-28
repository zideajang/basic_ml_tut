## 岭回归(Ridge Regression)
需要看一下标准方程方法
$$ w = (X^TX)^{-1}X^Ty $$

如果训练样本 m 数量少，甚至少于样本特征点 n，这样将导致数据矩阵$(X^TX)$ 不是满秩矩阵，而无法求逆。

为了解决这个问题，统计学家引入了**岭回归**的概念

$$ w = (X^TX + \lambda I)^{-1}X^Ty $$

$\lambda$为领系数，I为单位矩阵(对角线上全为 1 其他元素为0)

$$ J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_{\theta}(x_i) - y_i)^2 + \lambda \sum_i^n \theta_i^2 $$

$$ J(\theta) $$