### 对称阵的特征向量
### 主成分分析

样本为

$$A = \left( \begin{matrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\ 
    a_{21} & a_{22} & \cdots & a_{2n} \\ 
    \vdots & \vdots & \vdots & \vdots \\
    a_{m-1,1} & a_{m-1,2} & \cdots & a_{m-1,n} \\ 
    a_{m,1} & a_{m,2} & \cdots & a_{m,n} \\ 
\end{matrix} \right) = \left( \begin{matrix}
    a_1^T \\
    a_2^T \\
    \vdots \\
    a_{m-1}^T \\
    a_m^T \\
\end{matrix} \right)$$

$$A \cdot \mu = \left( \begin{matrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\ 
    a_{21} & a_{22} & \cdots & a_{2n} \\ 
    \vdots & \vdots & \vdots & \vdots \\
    a_{m-1,1} & a_{m-1,2} & \cdots & a_{m-1,n} \\ 
    a_{m,1} & a_{m,2} & \cdots & a_{m,n} \\ 
\end{matrix} \right)  \cdot \mu  = \left( \begin{matrix}
    a_1^T \\
    a_2^T \\
    \vdots \\
    a_{m-1}^T \\
    a_m^T \\
\end{matrix} \right) \cdot \mu  =  \left( \begin{matrix}
    a_1^T \cdot \mu \\
    a_2^T \cdot \mu \\
    \vdots \\
    a_{m-1}^T \cdot \mu \\
    a_m^T \cdot \mu \\
\end{matrix} \right)$$

### 计算向量的方差
$$Var(A \cdot \mu) = Var(a_1^T,a_2^T,\cdots,a_{m-1}^T,a_m^T)^T = \frac{1}{m-1} \sum_{i=1}^m(a_i^T - E)^2$$

$$(a_1^T,a_2^T,\cdots,a_{m-1}^Ta_m^T) \cdot (a_1^T,a_2^T,\cdots,a_{m-1}^Ta_m^T)^T = (A\mu)^T(A\mu) = \mu^T A^T A \mu$$
如果
$$\mu^T A^T A \mu = \lambda \Rightarrow \mu \mu^T A^T A \mu = \mu \lambda \Rightarrow A^TA \mu = \lambda \mu $$

PCA 的两个特征向量
- 特征提取
- 数据压缩(去噪)

难于解释降维后数据特征代表意义

向量再考察
- 期望
    - 离散型 $E(X) = \sum_i x_i p_i$
    - 连续型 $E(X) = \int_{ - \infty}^{\infty} x f(x) dx $
- 期望的性质
    - 无条件成立 
        $E(kX) = kE(X)$  $E(X + Y) = E(X) + E(Y)$ 
    - 若 X 和 Y 相互独立
        E(XY) = E(X)E(Y) 反之不成立
- 方差
    - 定义 $Var(X) = E([X - E(X)]^2)$
    - 无条件成立
        $Var(c) = 0 $ $Var(X + c) = Var(X)$$Var(kX) = k^2Var(X)$
    - X 和 Y 独立
        $$Var(X + Y) = Var(X) + Var(Y)$$
- 协方差
- 协方差矩阵
- 相关系数
- 向量间的