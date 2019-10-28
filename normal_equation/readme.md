## 标准方程法

损失函数
$$ J(\theta) = a\theta^2 + b\theta + c $$

$$ J(\theta_0, \theta_1, \cdots \theta_m) = \frac{1}{2m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2$$
$$ \frac{\partial J(\theta)}{\partial \theta_j} = \cdots = 0$$
通过数学手段来进行求导，求解处理损失函数全局最小值对应参数值。
求解: $\theta_0, \theta_1, \cdots \theta_n$

$$
X = \left[ 
    \begin{matrix}
        1 & 2104 & 5 & 1 & 45 \\
        1 & 1416 & 3 & 2 & 40 \\
        1 & 1534 & 3 & 2 & 30 \\
        1 & 852 & 2 & 1 & 36 
    \end{matrix}
    \right]
$$
这里损失函数可以用矩阵形式表达
$$ \sum_{i=1}^m(h_w(x^i) - y^i)^2 = (y- Xw)^T(y- Xw) $$
$Xw$ 进行乘积会得到 4x1 矩阵，所以 Xw 就等于预测值。感觉这里推导不难，只要稍微动手推导一下就可以得到损失函数表达式。

#### 分子布局(Numberator-layout)
分子为列向量或者分母为行向量
#### 分母布局(Numberator-layout)
分子为列行量或者分母为列向量

$$ \frac{\partial (y - Xw)^T(y- Xw) }{\partial w}$$
我们对上面推导矩阵方程进行求导

$$ \frac{\partial (y^Ty - y^TXw - w^TX^Ty + w^TX^TXw) }{\partial w}$$
先将分母部分进行展开，这个不难可以自己动手展开一下。
$$ \frac{\partial y^Ty}{\partial w} - \frac{\partial y^TXw}{\partial w}  - \frac{\partial w^TX^Ty}{\partial w}  + \frac{\partial w^TX^TXw}{\partial w} $$

$\frac{\partial y^Ty}{\partial w} =0$
这个可以查表进行求导，w 是列向量所以是 **分母布局**

$ \frac{\partial y^TXw}{\partial w} = X^Ty$

$\frac{\partial w^TX^Ty}{\partial w} = \frac{\partial (w^TX^Ty)^T}{\partial w} = \frac{\partial y^TXw}{\partial w} = X^Ty$

$ \frac{\partial w^TX^TXw}{\partial w} = 2X^TXw$

$$ \frac{\partial y^Ty}{\partial w} - \frac{\partial y^TXw}{\partial w}  - \frac{\partial w^TX^Ty}{\partial w}  + \frac{\partial w^TX^TXw}{\partial w} = 0 - X^Ty - X^Ty + 2X^TXw $$

$$ -2X^Ty + 2X^TXw = 0 $$
$$ X^TXw = X^Ty $$
$$ (X^TX)^{-1}X^TXw =(X^TX)^{-1} X^Ty $$
$$ w = (X^TX)^{-1}X^Ty $$
$(X^TX)^{-1}$ 是 $(X^TX)$ 的逆矩阵
梯度
### 矩阵不可逆的情况
1. 线性相关的特征(多重共线性)
2. 特征数据太多(样本数$m \le n$ 特征数量 n) 
### 梯度下降法 VS 标准方程法
#### 梯度下降法
- 缺点
需要选择合适的 **学习率**，迭代很多周期，只能得到最优解
- 优点
可以应对非常多的特征值
#### 标准方程法
- 缺点
计算$(X^TX)^{-1} $ 时间复杂度大约是$O(n^3)$
n 是特征数量
- 优点
