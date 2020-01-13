## PCA 主成分分析
#### 知识背景
- 协方差
- SVD 分解
- 特征值与特征向量


#### PCA算法两种实现方法
- 基于特征值分解协方差矩阵实现PCA算法
- 基于SVD分解协方差矩阵实现PCA算法


### SVD 计算
有关 SVD 应用很广泛
奇异值分解是一个能适用于任意矩阵的一种分解的方法，对于任意矩阵A总是存在一个奇异值分解：
$$ \left( \begin{matrix}
    0 & 1 \\
    1 & 1 \\
    1 & 0 
\end{matrix} \right)$$

$$A^TA = \left( \begin{matrix}
    0 & 1 & 1 \\
    1 & 1 & 0 \\
\end{matrix} \right) \left(\begin{matrix}
    0 & 1 \\
    1 & 1 \\
    1 & 0 
\end{matrix} \right) = \left(\begin{matrix}
    2 & 1 \\
    1 & 2 
\end{matrix} \right)$$

$$AA^T =  \left(\begin{matrix}
    0 & 1 \\
    1 & 1 \\
    1 & 0 
\end{matrix} \right) \left( \begin{matrix}
    0 & 1 & 1 \\
    1 & 1 & 0 \\
\end{matrix} \right) = \left(\begin{matrix}
    1 & 1 & 0 \\
    1 & 2 & 1 \\
    0 & 1 & 1 
\end{matrix} \right)$$