## 决策树算法
信息熵：随机变量的不确定性
- 熵越大，数据不确定性越高
- 熵越小，数据不确定性越低

$$ H = - \sum_{i=1}^k p_k \log (p_k)$$

$$ x \{ \frac{1}{3},\frac{1}{3},\frac{1}{3} \}$$
$$ - \frac{1}{3} \log(\frac{1}{3}) - \frac{1}{3} \log(\frac{1}{3}) - \frac{1}{3} \log(\frac{1}{3}) \approx 1.09$$
$$ x \{ \frac{1}{10},\frac{2}{10},\frac{7}{10} \}$$
$$ - \frac{1}{10} \log \frac{1}{10 } - \frac{2}{10} \log \frac{2}{10} - \frac{7}{10} \log \frac{7}{10} \approx 0.8018$$

我们看 0.8 要优于 1.09 也因为 0.8 不确定性较小，例如语言中只有一个字母那么因为他是确定的，所以 $log 1 = 0$
假设
### 决策树的概率
### 决策树的算法
#### 纯度计算公式
### 信息熵概念
我们知道信息是用来消除不确定性的东西。对于同一句话信息量是不同的。
1. 信息量和事件发生的概率有关，事件发生的概率越低，传递信息量越大
2. 信息量应当是非负的，必然发生的事件的信息量为零
3. 两个事件的信息量可以相加，并且两个独立事件的联合信息量应该是他们各自信息量的和

- 信息量 $h(x) = \log_a \frac{1}{p_x} = - \log_a P_x$
这个公式是完全满足上面 3 个条件。在信息学用 0 和 1 讨论所有 a 一般采用 2 而不是 10 
- 信息熵 $Ent(D) = \sum_{k=1}^m p_k \log_2 \frac{1}{p_k} = - \sum_{k=1}^m p_k log_2 p_k$
- 联合信息量如果两个信息是独立，那么他们信息量是两个单独信息的求和。
    $ P(AB) = P(A)P(B) = f(P(AB)) = f(PA) + f(PB)$
  
信息熵是信息量的数学上的期望。信息熵就是信息量的不确定，不确定度一旦消除我们信息纯度就提高

通过决策树不断分解将信息熵降下来，j
### 从 ID3 到 C4.5

#### 求职问题
- 第一列表示待遇高低 1 表示高 0 表示低
- 标准第二列表示岗位是否有五险一金 1 表示有五险一金 0 如果没有公积金只有五险用 2 来表示
- 1 表示接受 offer 0 表示没有接受 offer
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

- 熵概念: 信息熵是用来描述信息的混乱程度或者信息的不确定度。同样信息之父在香农提出信息熵是可以反映集合的纯度，这是因为不确定度越大那么纯度就越低。我们通过决策树不断分解将信息熵降低下来。

我们要计算纯度也就是计算其不确定度，不确定度大其纯度就低。例如 A 和 B 两个事件的概率都是 50% 那么不确定度很好。我们可以根据纯度的算法来对决策树进行划分，还可以根据剪枝的策略对决策树进行划分。首先我们来介绍 ID3 ，随着 ID3 推测后大家关注这算法并不断扩展在此基础推出 ID4 ID5 后来因为名字都被占用才有了 C4.5 C5.0 所以这里大家不用奇怪为什么开始叫 ID3 后来又叫 C4.5

- 信息熵计算公式

$$ Ent(D) = \sum_{k=1}^m p_k \log_2 \frac{1}{p_k} = - \sum_{k=1}^m p_k \log_2 p_k$$


- 信息增益公式
$$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

- 信息增益率公式
$$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

#### 根据上面示例我们进行计算
我们以招聘为例，从样本看接受 offer 和没有接受 offer 各占 50% 
我们接受和没有接受 offer 概率分别是
$$ p_1 = \frac{1}{2}, p_2 = \frac{1}{2}$$
那么我们根据信息熵公式来计算，所以根节点信息熵为 1
$$ - ( \frac{1}{2} \log_2 \frac{1}{2} +  \frac{1}{2} \log_2 \frac{1}{2}) = 1$$

```
-(0.5 * math.log(0.5,2) + 0.5 * math.log(0.5,2))
```

下面公式表示基于 C 条件（也就是条件熵)划分后得到两个节点的信息熵是多少

$$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

我们希望分支之后信息熵是降低的，我么选择条件 C 后这两个节点，我们回去查表看看高薪岗位的有几个，一共有 6 个岗位。那么是高薪招聘到人是 5 个，

 |  | 招聘到人 | 没有招聘到人 |
| ------ | ------ | ------ |
| D1 | $\frac{5}{6}$ | $\frac{1}{6}$ |
| D2 | 0 | 1 |
薪资高岗位为 6 个，而薪资低为 4 个那么在薪资高岗位招聘到人情况如表中第一行，也就是6 岗位招聘到了 5 人，有一个高但是没有招聘到人。而薪资低岗位全军覆没没有招聘到一个人，这说明大家还是很看重薪酬的。

好我们就先算 $D_1$ 情况下信息熵之和然后在计算 $D_2$ 情况下信息熵之和。

表示属于高薪节点(D1)的信息熵 
$$ Ent(D_1) = - (\frac{5}{6} \log \frac{5}{6} + \frac{1}{6} \log \frac{1}{6} ) \approx 0.65$$
表示不是高薪节点(D2)的信息熵
$$ Ent(D_2) = 1 * \log_2^1 + 0 * \log_2^0 = 0 $$
然后我们用根节点的信息熵1 减去D1 和 D2 信息熵来算信息增益，信息熵越大说明集合不确定性越高，那么我们由于信息增益是划分前后信息熵之差，这个值表示信息熵降低幅度，所以信息增益越大说明划分条件越好。

$$ 1 - (\frac{6}{10} * 0.65 + \frac{4}{10} * 0) = 0.61 $$


表示划分后信息熵为，信息增益越大越好，使用信息熵天生对多路分类进行偏好。
```
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
我们在考虑第中情况按
- 有保险情况 1 $\frac{3}{10}$
- 没有保险情况 0  $ \frac{4}{10}$
- 有保险没有公积金  2 $\frac{3}{10}$ 


|  分类 | 招聘到人 | 没有招聘到人 |
| ------ | ------ | ------ |
| c0 | 0 | 1 |
| c1 | 1 | 0 |
| c2 | $\frac{2}{3}$ | $\frac{1}{3}$ |

$$Ent(C_0) = 0 * \log_2^1 + 1 * \log_2^0 = 0  $$
$$ Ent(C_1) = 1 * \log_2^1 + 0 * \log_2^1 = 0 $$
$$ Ent(C_2) = - (\frac{2}{3} \log \frac{2}{3} + \frac{1}{3} \log \frac{1}{3} ) \approx 0.637$$

$$ 1 - \frac{3}{10} \times 0.637 = 0.36 $$

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
##### 回归树

$$V(D) = \frac{1}{N-1} \sum_{i=1}^N(y_i(D) - \overline{y}(D))^2$$
$$ \Delta D = $$

### 决策树的评估
