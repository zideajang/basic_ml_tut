$$L(w,b,a) = \frac{1}{2} ||w||^2 - \sum_{i=1}^n \alpha_i (y_i(w^T \cdot \Phi(x_i) + b) - 1)$$
$$\min_{w,b} \max_{\aleph} L(w,b,a)$$
原始问题的对偶问题，是极大和极小问题
$$ \max_{\aleph} \min_{w,b} L(w,b,a)$$

最求其最小值，然后我们 w 和 b 进行求偏导数
$$\frac{\partial L}{\partial w} \Rightarrow w - \sum_{i=1}^n \alpha_i y_i \Phi(x_i)= 0$$
$$ \frac{\partial L}{\partial b} \Rightarrow \sum_{i=1}^n \alpha_i y_i = 0 $$

$$w = \sum_{i=1}^n \alpha_i y_i \Phi(x_i)$$
其中$\alpha_i \ge 0 $

### 计算拉格朗日函数的对偶函数
$$\begin{aligned}
    L(w,b,a) = \frac{1}{2} ||w||^2 - \sum_{i=1}^n \alpha_i (y_i(w^T \cdot \Phi(x_i) + b) - 1) \\
    = \frac{1}{2} w^Tw - w^T\sum_{i=1}^n \alpha_i y_i \Phi(x_i) - b \sum_{i=1}^n \alpha_iy_i + \sum_{i=1}^n \alpha_i \\
    = \frac{1}{2} w^T \sum_{i=1}^n \alpha_i y_i \Phi(x_i) - w^T\sum_{i=1}^n \alpha_i y_i \Phi(x_i) - b \cdot 0 + \sum_{i=1}^n \alpha_i \\
    = \sum_{i=1}^n \alpha_i - \frac{1}{2} \left( \sum_{i=1}^n \alpha_i y_i \Phi(x_i) \right)^T \sum_{i=1}^n \alpha_i y_i \Phi(x_i)\\
    = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j =1}^n \alpha_i \alpha_j y_i y_j \Phi^T(x_i)\Phi(x_i)
\end{aligned}$$

$$a^* = \arg \max_{\alpha} \left( \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j =1}^n \alpha_i \alpha_j y_i y_j \Phi^T(x_i)\Phi(x_i) \right)$$