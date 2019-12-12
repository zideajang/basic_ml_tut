## 极大似然估计(ML 估计)
ML 是 maximum likehood 的缩写，如果我们将模型描述为一个概率模型，那么我们就希望得到在参数$\theta$ 能够使训练集输入时，输出概率达到极大。那么什么又是似然呢?
$$P(X|\theta) = \prod_{i=1}^N p(x_i|\theta)$$

$$\hat{\theta} = \arg \max_{\theta} p(X|\theta) = \arg \max_{\theta} \prod_{i=1}^N p(x_i|\theta) $$

假设暗箱中有 2 红球和篮球两个球，有放回抽取小球，结果是1红1蓝，根据抽取结果进行判断暗箱中是 1 红 1蓝结果比较合理，这个估计就是最大似然估计。

$$\begin{aligned}
    p(X|\theta) = \theta^{x_1 + x_2}(1-\theta)^{2-x_1-x_2} 
\end{aligned}$$
- 这里$\theta$ 是估计红球概率
- $x_i$ 表示每一次取球结果，如果取中红球就是 1 否则为 0

$$\begin{aligned}
    p(X|\theta) = \theta^{x_1 + x_2}(1-\theta)^{2-x_1-x_2} \\
    \Rightarrow \ln P(X|\theta) = (x_1 + x_2) \ln \theta + (2-x_1-x_2)\ln (1-\theta)
\end{aligned}$$
求极值值就是转化为求导问题，那么就是对上面式求导

$$\begin{aligned}
    \Rightarrow \frac{\partial p}{\partial \theta} = \frac{x_1 + x_2}{\theta} - \frac{2 - x_1 - x_2}{1 - \theta} \\
    \frac{\partial p}{\partial \theta} = 0 \Rightarrow \theta = \frac{x_1 + x_2}{2}
\end{aligned}$$

$$\begin{cases}
    x_1 = 1 & x_2 = 0 & \hat{\theta} = 0.5 \\
    x_1 = 1 & x_2 = 1 & \hat{\theta} = 1 \\
    x_1 = 0 & x_2 = 0 & \hat{\theta} = 0 \\
\end{cases}$$

这里应该不难理解如果两次取出分别是红球和篮球那么也就是 $x_1,x_2$ 都是 1 和 0 那么$x_1 + x_2 = 1$ 从而$\theta = 0.5$ 也就是取出红球概率值为 0.5，那么根据样本从而判断箱子内为红球和篮球各一个。