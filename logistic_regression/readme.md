

### 曲线回归
- 
$$ y = ln(ax + b) $$
$$ y \prime = e^y $$
$$ e^y = ax + b = y \prime $$

$$ y = \frac{a}{x} + b $$
$$ y = \frac{1}{1 + e(-ax + b))} $$

$$ y = ax + bx^2 + c $$
$$ x_1 = x $$
$$ x_2 = x^2 $$
$$ y =  ax_1 + bx_2 + c $$
面对这些曲线我们就无法用线性模型

$$ w^* = \arg \min $$

### 正则化(Regulariztion)
为什么需要线性回归，我们先复习一下线性回归最小二乘法
$$ J(\theta) = \frac{1}{2m}[\sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2] $$
$$ J(\theta) = \frac{1}{2m}[\sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n |\theta_j|] $$