$$ \theta^* = arg \min_{\theta} L(\theta) $$
- L:损失函数
- $\theta$: 参数
- 参数 $\theta$ 假设有两个参数${\theta_1, \theta_2}$
- 然后随机选取一组参数
$$ \theta^0 = \left[ \begin{matrix}
    \theta_1^0 \\
    \theta_2^0
\end{matrix}
\right]
$$
- 计算他们对于 L 的偏微分
$$ \theta^1 = \left[ \begin{matrix}
    \theta_1^0 \\
    \theta_2^0
\end{matrix}
\right] - \eta \left[ \begin{matrix}
    \frac{\partial(\theta_1^0 )}{\partial \theta} \\
    \frac{\partial(\theta_2^0 )}{\partial \theta} 
\end{matrix}
\right]
$$

$$ \theta^2 = \left[ \begin{matrix}
    \theta_1^1 \\
    \theta_2^1
\end{matrix}
\right] - \eta \left[ \begin{matrix}
    \frac{\partial(\theta_1^1 )}{\partial \theta} \\
    \frac{\partial(\theta_2^1 )}{\partial \theta} 
\end{matrix}
\right]
$$

### 调整学习(learniing rates)

$$\theta^i = \theta^{i-1} - \eta  L(\theta(\theta^{i-1}))$$

