## 决策树算法
### 决策树的概率
### 决策树的算法
#### 纯度计算公式
#### 从 ID3 到 C4.5
#### 问题
```python
1. 1 1 => 1
2. 1 2 => 1
3. 0 0 => 0
4. 0 0 => 0
5. 1 1 => 1
6. 0 0 => 0
7. 1 2 => 1
8. 1 1 => 1
9. 0 2 => 0
10. 1 0 => 0

```

- 熵概念: 信息熵是用来描述信息的混乱程度或者信息的不确定度。信息熵是可以反映集合的纯度
- 算法目的通过不断决策树来将信息熵降低下来
- 信息熵 $$ Ent(D) = \sum_{k=1}^m p_k \log_2 \frac{1}{p_k} = - \sum_{k=1}^m p_k \log_2 p_k$$
$$ p_1 = \frac{1}{2}, p_2 = \frac{1}{2}$$
$$ - ( \frac{1}{2} \log_2 \frac{1}{2} +  \frac{1}{2} \log_2 \frac{1}{2}) = 1$$
- 信息增益 $$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

- 信息增益率
$$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$
划分条件为待遇
$$ Ent(D_1) = - (\frac{5}{6} \log \frac{5}{6} + \frac{1}{6} \log \frac{1}{6} ) \approx 0.65$$

$$ Ent(D_2) = 1 * \log_2^1 + 0 * \log_2^0 = 0 $$

$$ 1 - (\frac{6}{10} * 0.65 + \frac{4}{10} * 0) = 0.61 $$
表示划分后信息熵为，信息增益越大越好，使用信息熵天生对多路分类进行偏好
- 信息增益率
$$ Gain_ratio(D,C) = \frac{Gain(D,C)}{Ent(C)} $$

$$ Ent(C) = - \sum_{i=1}^k \frac{N(D_i)}{N} \log_2 \frac{N(D_i)}{N} $$

对于多路划分乘以代价
$$ Gain(D C_1) = 0.61, Gain(D C_2) = 0.72$$
$$ Ent(C_1) = - (\frac{6}{10} \log \frac{6}{10} + \frac{4}{10} \log \frac{4}{10}) = = 0.971$$
$$ Ent(C_2) = 1.571$$

$$ Gain_ratio = 0.63  $$

#### CART 算法
CART 即分类与回归树(Classfication And Regression Tree) CART 算法与 ID3 算法思路是相似的，但是在具体实现和场景上略有不同。
- CART 不仅可以处理分类问题还可以处理连续问题
- CART 算法采用基尼指数(分类树)以及方差（会归树）作为纯度的度量，而 ID3 系列算法采用信息熵作为纯度
- CART 算法只能二叉树
##### 分类树
$$ Gini(D) = \sum_{k=1}^m p_k(1 - p_k) = 1 - \sum_{k=1}^m p_k^2$$

$$ \Delta Gini = Gini(D) -   \sum_{i=1}^2 \frac{N(D_i)}{N} Gini(D_i)$$
##### 会归树
