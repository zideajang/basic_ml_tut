信息熵在语言，我们可以通过语言来解释一下信息熵，中文的信息熵

```
>>> -((1/3)* math.log(1/3)) * 3
1.0986122886681096
>>> -((1/3)* math.log(1/3)) * 5
1.8310204811135162
>>>
```

在英文中只有 26 字母，通过26字母排列组合

$$\frac{1}{3},\frac{1}{3},\frac{1}{3}\frac{1}{3}\frac{1}{3}$$

$$\frac{1}{3},\frac{1}{3},\frac{1}{3}$$
我们知道中文中信息熵大概是 9.2 而英文信息行,
在英文中即使缺失几个字母，并不影响我们读懂英文句子，而中文却不同，在中文中我们
![dt_01.jpg](https://upload-images.jianshu.io/upload_images/8207483-29675bd966b05172.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 决策树(Decisino Tree)
决策树(Decision Tree）是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。
### 前言
我们将讨论可以用于**分类**和**回归**任务的简单、非线性的模型——**决策树**。个人认为决策树相对于其他机器学习模型更容易理解，符合我们人类思考和对事物进行分类判断的逻辑。我们在现实生活中就是这样一步一步做出决策的。例如找对象、找工作都会不自觉用到决策树算法来做出决定我们的决定。有人可能认为现在是深度学习的天下，决策树是不是已经过时了没有立足之地了。虽然在某些应用已经打上深度神经网的标签，但是在一些领域中可能更适合选择决策树而非深度神经网模型。
### 决策树
![dt_011.jpeg](https://upload-images.jianshu.io/upload_images/8207483-99aa3dc5a7bcc4d9.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

决策树是一个**非线性**分类器,给出以一种分割数据的规则来将数据切分为 $N_1$ 和 $N_2$ 两份数据。既然这里提到了线性和非线性模型，我们简单介绍一下什么是线性模型和什么是非线性模型，以及两者的区别。
![dt_03.jpg](https://upload-images.jianshu.io/upload_images/8207483-353e05689bd393ee.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 线性模型
在统计意义上，如果一个回归等式是线性的，其参数就必须也是线性的。如果相对于参数是线性，那么即使性对于样本变量的特征是二次方或者多次方，这个回归模型也是线性的。
$$ y = \theta_0 + \theta_1x_1 + \theta_2x_2^2$$
#### 非线性模型
最简单的判断一个模型是不是非线性，就是关注非线性本身，判断其参数是不是非线性的。
$$ y = \theta_0 + \theta_1 x_1 + \theta_2^2 x_2 $$

- $x_1$ 和 $x_2$ 不是独立，在不同区域内 $x_1$ 和 $x_2$ 关系不同。

#### 线性模型和非线性模型的区别
在机器学习的回归问题中，**线性模型**和**非线性模型**都可以去对曲线进行建模，所有有许多人误认为线性模型无法对曲线进行建模。其实，线性模型和非线性模型的区别并不在于能不能去拟合曲线。
1. 线性模型可以用曲线拟合样本，但是分类的决策边界一定是直线(如 logistics 模型)
2. 区分是否为线性模型，主要是看一个乘法式子中自变量 $x$ 前的系数 $\theta$ ，应该是说 $x$ 只被一个 $\theta$ 影响，那么此模型为线性模型。
3. 画出 $y$ 和 $x$ 是曲线关系，但是他们是线性模型，因为 $theta_1x_1$ 中可以观察到 $x_1$ 只被一个 $w_1$ 影响
   $$ y = \frac{1}{e^{(\theta_0,+ \theta_1x_1 + \theta_2x_2)}}$$
4. 当(如下面模型中) $x_1$ 不仅仅被参数$\theta_1$ 影响，而且还被$\theta_5$ 影响，自变量 $x$ 被两个以上的参数影响，那么模型就是非线性模型
    $$ y = \frac{1}{1 + \theta_5 e^{(\theta_0,+ \theta_1x_1 + \theta_2x_2)}}$$

![dt_08.jpg](https://upload-images.jianshu.io/upload_images/8207483-f155693ba75b931f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 看一个例子
1. 看待遇
2. 看是否双休
3. 看是否出差
4. 看是否
5. 看是否有发展
通过系列问题我们衡量最终结果是去还是不去这家公司。这系列问题就是



#### 决策树的结构
![dt_09.jpg](https://upload-images.jianshu.io/upload_images/8207483-17e85d2b41255fd9.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

决策树算法是用树状结构表示数据分类的结果
- 根节点(root node)
- 非叶子节点(internal node)
- 叶子节点(leaf node)每一个叶子节点都有一个分类值
- 分支(branch)
![dt_07.jpg](https://upload-images.jianshu.io/upload_images/8207483-4b80d252e374b616.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在开始之前，我们看一些重要概念以及公式推导。
### 信息熵
熵概念最早是在物理热力学中提出概念后来也被在统计学中使用，表示提出了**信息熵**的概念，用以度量信息的不确定性。度量数据的纯度的指标这里用**信息熵**。一个事件发生概率与这个事件信息的关联，完全确定的也就是一定会发生，那么信息就会为 0。既然我们知道信息熵和概率的关系我们下面来推导一下**信息熵**计算公式
$H(x) H(y)$ 表示事件发生不确定性，也就是事件信息熵，他们分别表示x 和 y 的信息熵。

例如说太阳从东方升起那么这句话就是信息量$H(x)$为 0 的话，这里信息量是指事件本身，也就是小概率发生了那么他信息量就是很大。那么也就是说概率越大其携带的信息量就越少。

![dt_10.jpg](https://upload-images.jianshu.io/upload_images/8207483-10c1a5c609d2357e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们通过下面实际例子来说说信息熵，下面有 A 和 B 两个集合分别是不同类别的元素，在 A集合包含各种不同种类水果的元素，而相对于 A 集合 B 集合包含了属于较少类别的元素。


![分类](https://upload-images.jianshu.io/upload_images/8207483-11abca1629b0a5cd.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ A \{x_{apple},x_{banana},x_{strawberry},x_{banana},x_{orange} \dots x_{grape}, x_{orange} \} $$
在 A 集合中有属于 banana apple straberry orange grape 等不同类别的元素
$$ B \{x_{apple},x_{banana},x_{apple},x_{banana} \dots ,x_{banana} ,x_{banana}  \} $$
而在 B 集合中仅包含较少类别的水果

不难看出 A 集合和 B 集合，A 集合中种类中水果品种类别出现比较多，说明 A 集合比 B 集合的信息熵大。也就是所有 A 集合类别比较高。
下面图为$ -ln(x)$
![屏幕快照 2019-11-10 下午4.56.16.png](https://upload-images.jianshu.io/upload_images/8207483-73032c11e7f169dc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


ln(x) 是增函数，那么 -ln(x) 就是一个减函数。所有在 x = 1 处 -ln(x) 为 0 而在 x = 0 时候 -ln(x) 却是无穷大

下面公式就是信息熵的公式
- 在离散情况下信息熵公式
$$ H(x) = -\sum_{i=1}^n P(x_i) \log P(x_i) $$

- 在连续函数情况下信息熵的公式
$$ H(P) = - \int p(x) lnp(x) dx$$

#### 联合概率
假设 x 和 y 是独立的事件，可以那么 x 和 y 两个事件的联合概率，
$$ P(x,y) = P(x)P(y)$$
我们希望信息量进行加减而不是乘法，如果将乘积变成去和的方式就会想到 log 来解决对于公式两边都取 log
$$ \log P(x,y) = \log P(x)P(y) = \log P(x) + \log P(y) $$


### 条件熵
用于在有信息情况的信息，
在 x 给定掉件 y 的信息熵，(x,y) 发生所包含的熵，减去 X 单独发生包含的熵:在 x 发生的前提下, y 发生**新**带来的熵。也就是如果已知 x 的条件下是否给 y 的熵带来变化。


$$ H(Y|X) = H(X,Y) - H(X) $$???
首先我们看一下熵的一些基本操作
- 在 x 和 y 都是相互独立
$$ H(Y|X) = H(x) + H(y)$$
- 在 x 和 y 并非是相互独立
$$ H(Y|X) < H(x) + H(y)$$

我们这里对 $$H(x,y)$$ 进行简单推导
根据信息熵的概念我们可以推导下面公式
$$ H(x,y)=  - \sum_{x,y} P(x,y)\log(P(x,y)) $$
然后根据$P(x,y) = p(y|x)P(x)$ 带入下面等式就可以得到下面公式
$$ = -\sum_{x,y} P(x,y) \log (P(x)P(y|x)) $$
进一步推导为
$$ = - \sum_{x,y}  P(x,y)(\log P(x) + \log P(y|x)) $$
$$ = - \sum_{x,y}  P(x,y) \log P(x) - \sum_{x,y} P(x,y) \ln P(y|x)) $$
因为这里我们只对 x 进行积分(求和)所有可以在前半部化简为
$$ = - \sum_{x}  P(x) \ln P(x) - \sum_{x,y} P(x,y) \ln P(y|x)) \tag{1}$$
$$ = H(x) - \sum_{x,y} P(x,y) \ln P(y|x)) $$
<!-- 到现在为止我们整理一下思路，我们???
$$ \begin{cases}
    H(y|x) = H(x,y) - H(x) \\
    I(x,y) = H(x,y) - H(x) - H(y) \tag{1}
\end{cases} $$ -->
### 互信息
在概率论和信息论中，两个随机变量的**互信息**（Mutual Information，简称MI）或转移信息（transinformation）是变量间相互依赖性的量度。不同于相关系数，互信息并不局限于实值随机变量，它更加一般且决定着联合分布 p(X,Y) 和分解的边缘分布的乘积 p(X)p(Y) 的相似程度。互信息(Mutual Information)是度量两个事件集合之间的相关性(mutual dependence)。互信息是点间互信息（PMI）的期望值。互信息最常用的单位是bit。

**互信息**指的是两个随机变量之间的关联程度，即给定一个随机变量后，另一个随机变量不确定性的削弱程度。只有真正理解好了**互信息**我们才能够理解在决策树背后是如何通过节点来降低数据的信息熵从而达到分类和回归的效果。

互信息，也就是度量信息，其实互信息就是度量两个概率，那么也就是独立情况下互信息为 0，如果不独立就是大于 0 的某个数，这个数表示两个概率距离。

$$ I(x,y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} $$
概率分布
$$I(x,y) = \int_Y \int_X P(x,y) \log \left( \frac{P(x,y)}{P(x)P(y)} \right) dx dy$$

![dt_11.jpeg](https://upload-images.jianshu.io/upload_images/8207483-cfd1ceffe6b35d02.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$I(x,y) = H(x) - H(x|y) $$
$$I(x,y) = H(y) - H(x|y) $$
$$I(x,y) = H(x) + H(y) - H(x,y)$$

这是因为我们看一下下面推导就可以明白为什么$I(x,y)$ 可以作为度量两个不相互独立之间熵的影响，
$$
    \begin{aligned}
        H(x,y) = H(x)+ H(y) \\
        H(x,y) < H(x) + H(y) 
    \end{aligned}
$$

$$ H(y|x) = H(x,y) - H(x) = -\sum_{x,y} P(x,y) \log P(y|x) $$
等效证明互信息是大于等于 0，今天证明的确有点烧脑，不过对我们能够理解决策树中一些算法还是有帮助的。

$$ I(x,y) = H(x,y) - H(x) - H(y) $$
只要证明了$ I(x,y) \le 0 $ 也就是说明$H(x,y) \le H(x) + H(y)$,也就证明了 
$$ H(Y|X) - H(Y) $$
$$ = - (\sum_{x,y} p(x,y) \log P(y|x)) + \sum_{x,y} P(x,y) \log P(y) $$


$$ = -\left[ \sum_{x,y} P(x,y)(\log \frac{P(x,y)}{P(x)} - \log P(y)) \right]$$
$$ = -\sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} $$

$$H(Y|X) - H(X) = - I(X,Y)$$

### 构建决策树
我们有了上面的基础知识，了解了什么是条件熵和互信息，表示我们可以通过给定特征来降低信息熵，我们回到决策树的问题上来看一看。
我们如何设计一颗决策树，选择那给个条件作为根节点，有了上面知识，我们不难使用公式来判断可以快速降低数据的信息熵的节点作为首要节点。
![dt_016.jpg](https://upload-images.jianshu.io/upload_images/8207483-ae3e17f56ead485a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 从 ID3 到 C4.5
#### ID3 定义
**ID3 算法**的核心是在决策树各个子节点上应用信息增益准则选择特征，递归的构建决策树，具体方法是:从根节点开始，对节点计算所有可能的特征的**信息增益**，选择信息增益最大的特征作为节点的特征，由该特征的不同取值建立子节点；再对子节点递归调用以上方法，构建决策树。

我们通过一个具体实例来讲解如何通过 ID3 算法来选择节点。

#### 求职问题

![dt_015.jpg](https://upload-images.jianshu.io/upload_images/8207483-fa44bbf400979803.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 第一列表示待遇高低 1 表示高 0 表示低
- 标准第二列表示岗位是否有五险一金 1 表示有五险一金 0 如果没有公积金只有五险用 2 来表示
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
先复习一下什么是信息熵
- 熵概念: 信息熵是用来描述信息的混乱程度或者信息的不确定度。信息熵是可以反映集合的纯度

我们要计算纯度也就是计算其不确定度，不确定度大其纯度就低。例如 A 和 B 两个事件的概率都是 30% 50%，A 事件的不确定性(也就是信息熵) 要大 。算法目的通过不断决策树来将信息熵降低下来
我们先将公式推出。

$$ Ent(D) = \sum_{k=1}^m p_k \log_2 \frac{1}{p_k} = - \sum_{k=1}^m p_k \log_2 p_k$$


- 信息增益 $$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

- 信息增益率
$$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

#### 分析实例
我们以招聘为例，从样本看接受 offer 和没有接受 offer 各占 50% 所以他们概率分别是

$$  \begin{cases}
    P_{recive} = \frac{1}{2} \\
    P_{refuse} = \frac{1}{2}
\end{cases} $$
$$ - ( \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2}) = 1$$

```
-(0.5 * math.log(0.5,2) + 0.5 * math.log(0.5,2))
```

#### 划分条件为待遇情况
我们希望分支之后信息熵是降低的，我么选择条件 C 后这两个节点，我们回去查表看看高薪岗位的有几个，一共有 6 个岗位。那么是高薪招聘到人是 5 

 |  | 招聘到人 | 没有招聘到人 |
| ------ | ------ | ------ |
| D1 | $\frac{5}{6}$ | $\frac{1}{6}$ |
| D2 | 0 | 1 |

$$ Ent(D_1) = - (\frac{5}{6} \log \frac{5}{6} + \frac{1}{6} \log \frac{1}{6} ) \approx 0.65$$

```
-(5/6.0* math.log((5/6.0),2)) - 1/6.0 * math.log((1/6.0),2)
```
$$ Ent(D_2) = 1 * \log_2^1 + 0 * \log_2^0 = 0 $$

$$ 1 - (\frac{6}{10} * 0.65 + \frac{4}{10} * 0) = 0.61 $$

| 分类  | 数量  | 比例  |
|---|---|---|
| 0  |  $\frac{4}{10}$ | $ 0 \rightarrow 4, 1 \rightarrow 0$ |
| 1  |  $\frac{3}{10}$ | $ 0 \rightarrow 0, 1 \rightarrow 3$ |
| 2  |  $\frac{3}{10}$ | $ 0 \rightarrow 1, 1 \rightarrow 2$ |
信息增益越大越好
$$ \frac{3}{10} \times \left( -\frac{1}{3} \log \frac{1}{3} - \frac{2}{3} \log \frac{2}{3} \right) = 0.27 $$

$$ 1 - 0.27 = 0.73 $$

```
>>> -1/3.0 * math.log(1/3.0,2) - 2/3.0 * math.log(2/3.0,2)
0.9182958340544896
>>> 0.9182958340544896 * 0.3
0.27548875021634683
>>> 1 - 0.28
0.72
```

在 ID3 我们根据信息增益(越大越好)来选择划分条件，但是有一个问题有关是否高薪我们有两种结果为是高薪或者不是高薪，而对于待遇我们是有三种结果0 代表没有五险一金待遇，1 代表有五险没有一金，2 代表五险一金俱全的。

也就是划分后条件，我们再极端点按 id 进行划分那么每一都是确定所以所有信息熵都是 0 然后 1 - 0 就是 1 ，那么按 id 划分信息增益最高但是没有任何作用，使用信息熵天生对多路分类进行偏好所以。新来一个数据我们按 id 划分是无法预测新数据类别的，而且按 id 进行划分说明我们的模型比较复杂。复杂模型就很容易出现过拟合现象。为了解决这个问题引入**信息增益率***

$$- 1 * \ln 1 = 0 $$
$$ \sum_{i=1}^n (-ln1) = 0 $$

#### C4.5 算法

- 信息增益率
信息增益率是在信息增益基础上除以信息熵，也就是给我们增加分类追加了一个代价。也就是分类多了其分母就会变大从而将数值降低下来。
$$ Gain_ratio(D,C) = \frac{Gain(D,C)}{Ent(C)} $$

$$ Ent(C) = - \sum_{i=1}^k \frac{N(D_i)}{N} \log_2 \frac{N(D_i)}{N} $$

我们回头从新整理一下公式
- 划分条件为薪资
$$ Gain(D C_1) = 0.61$$
$$ Ent(C_1) = - (\frac{6}{10} \log \frac{6}{10} + \frac{4}{10} \log \frac{4}{10}) =  0.971$$
```
- (6/10.0 * math.log(6/10.0,2) + 4/10.0 * math.log(4/10.0,2))
```
$$ Gain_{ratio}(C_1) = 0.61 \div 0.971  = 0.63  $$
- 划分条件为福利待遇
$$Gain(D C_2) = 0.72$$
$$ Ent(C_2) = - (\frac{3}{10} \log \frac{3}{10} + \frac{3}{10} \log \frac{3}{10} + \frac{4}{10} \log \frac{4}{10}) = 1.571$$

```
-(3/10.0 * math.log(3/10.0,2) + 3/10.0*math.log(3/10.0,2)+4/10.0*math.log(4/10.0,2))
```

$$ Gain_{ratio}(C_2) = 0.72 \div 1.571  = 0.46  $$

如果我们仅信息增益就会选择C2(按待遇进行划分),而按信息增益率来计算我们就会选择C1(按薪酬划分）。

### CART 算法(Classification And Regression Tree)
CART 算法不但可以处理分类问题还可以处理回归问题。CART 算法与ID3 算法思路是相似的，但是具体实现上和应用场景略有不同。
CART 算法采用基尼指数（分类树）以及方差（回归树）作为纯度的度量，而ID3 系列算法采用信息熵作为纯度的度量。
CART 算法只能建立二叉树，而ID3系列算法能够建立多叉树。
#### 分类树
$$ Gini(D) = \sum_{k=1}^m p_k(1-p_k) = 1 - \sum_{k=1}^m p_k^2 $$
基尼指数越小越好
$$ \Delta Gini = Gini(D) - \sum_{i=1}^2 \frac{N(D_i)}{N} Gini(D_i)$$
我们用根节点的基尼指数减去按条件划分后基尼指数，也就是结果越大那么说明这个划分条件越好。
#### 回归树
对于连续型我们通过方差来考虑，方差越小纯度越高
$$ V(D) = \frac{1}{N-1} \sum_{i=1}^N(y_i(D) - \hat{y}(D))^2$$
同样我们用根节点方差减去划分后方差，结果越大说明划分条件越好
$$ \Delta V = V(D) - \sum_{i=1}^2 \frac{N(D_i)}{N} V(D_i)$$

### 优化以及总结
|   | C5.0 系列  | CART |
|---|---|---|
| 是否支持多路划分  |  多重(有多少类别数量就有多少)  | 二叉树|
| 连续目标|否|是|
| 决策树分割标准|信息增益(ID3)/信息增益率(C4.5)|Gini 系数(分类）/方差(回归树)|
| 剪枝|误差估计|误差代价——复杂度|
| 代价敏感矩阵|支持|支持|
| Boosting|支持|支持|

我们来思考一个问题，我们通过不断细分一定可以把数据划分为我们想要结果。但是这个树并不是我们想要的模样，因为上面我们已经知道了，复杂模型带来就是过拟合问题。那么我们如何评估一个决策树设计好坏呢?
所以我们就需要对决策树进行调整和删减不必要节点，这个过程我们称为剪枝，剪枝又分为预剪枝和后剪枝，在预剪枝很简单，必须保证分裂后节点必须保证有我们事先定义好的样本量，如果节点样本量少于指定的样本量，我们就不进行分支。这就是预剪枝，也就是生成节点如果不满足一定条件我们就将其减掉，这个过程就是预剪枝，有了预剪枝当然也就有后剪枝。
#### 后剪枝
- 我们用训练集构建决策树，然后我们用测试评估决策树，通过测试集后，假如父节点的误差大于分支后误差就说明划分是对的，相反就会减掉这个分支
- 而在C5.0会采用其他策略进行评估，C5.0 会考虑代价。
##### 延森不等式
![jensen-small.png](https://upload-images.jianshu.io/upload_images/8207483-300ae6cdca2368a8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
$$ \alpha_1 \log x_1 + \alpha_2 \log x_2 \le \log (\alpha_1x_1 + \alpha_2x_2)$$


再用其他方式下面我们来推导一下条件熵

$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} p(x) \log p(x)$$
有了这个不等式
$$ -\sum_{x,y}P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$
$$ =  \sum_{x,y}P(x,y) \log \frac{P(x)P(y)}{P(x,y)} $$
$$ =  \sum_{x,y} \log P(x,y) \frac{P(x)P(y)}{P(x,y)} $$
$$ \ge  \sum_{x,y} \log P(x)P(y) $$
上面不等式可以看出联合概率熵和要大于各种概率熵的和
$$ I(x,y) \ge 0 $$ 

$$ H(Y) - I(X,Y) $$
$$ = -\sum_y p(y) \log p(y) - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x),p(y)}$$
$$ = -\sum_y (\sum_x p(x,y))  \log p(y) - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x),p(y)}$$
$$ = -\sum_{x,y} p(x,y)  \log p(y) - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x),p(y)}$$

$$ - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)}$$
$$ - \sum_{x,y} p(x,y) \log p(y|x)$$
$$ H(Y|X) $$

### 推导互信息
$$ I(X,Y) = H(X) + H(Y) - H(X,Y) $$

#### 交叉熵
- 也称为相对熵、鉴别信息 Kullback 熵，Kullback-Leible 散度等
- 假设 p(x) q(x) 是 X 中取值的两个概率分布，则 p 对 q 的相对熵就是
$$ D(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = E_{p(x)} \log \frac{p(x)}{q(x)} $$
- 说明
    - 交叉熵可以度量两个随机变量的**距离**
    - 一般 $ D(p||q) D(q||p) $
    - $ D(p||q) \ge 0 D(q||p) \ge 0 $
$$ p(x) \approx q(x) $$
什么是交叉熵，P(x) 是未知的，我们实际上用q(x) 来逼近 p(x)
我们只能根据数据推测 P(x)

$$ H_{cross} = - \int P(x) \log q(x) dx $$
$$ H_{cross} \ge S = - \int p(x) \log P(x) dx $$

接下我们来一起证明一下
$$ H(q(x)) - H(p(x))$$
$$ = - \int p(x) \log q(x) dx + \int p(x) \log p(x) dx $$
$$ = - \int p(x) \log \frac{q(x)}{p(x)} dx \approx KL $$
$$ H(q(x)) - H(p(x))  $$
$$ -\log \int q(x) dx$$

我们从另一个角度进行推导
把所有的 y 加起来进行积分后积分掉 x,
$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} (\sum_y p(x,y)) \log p(x)$$
$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x,y} p(x,y) \log p(x)$$

$$ - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x,y} p(x,y) p(x) $$

$$ - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)}$$
$$ -\sum_{x,y} p(x,y) log p(y|x) $$

## 公式推导

----

### 信息熵公式的推导
学过概率后,如果两个事件 x 和 y 是独立，这个应该不难理解$$ P(x,y) = P(x) \cdot P(y)$$ 
首先我们已经知道了信息熵和概率关系是成反比，也就是说概率越大信息熵越小，这个应该也不难理解，如果一个一定会发生事，例如太阳从东方升起，这个事件给我们带来信息量非常小
![dt_05.jpeg](https://upload-images.jianshu.io/upload_images/8207483-486e514a2beeaf77.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们知道$\ln x$函数模样如下图，$\ln x$ 是一个增函数
![lnx](https://upload-images.jianshu.io/upload_images/8207483-5167cb7e309e678b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果对函数添加负号就变成减函数，而且在 0 - 1 区间函数值符号我们要求就是在 1 时函数值为 0 而接近 0 时为无穷大
![-lnx](https://upload-images.jianshu.io/upload_images/8207483-cfd0f1caf3c84759.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然后我们进步想象一下如果希望两个事件的信息量也是可以做加减
$$H(y|x) = H(x,y) - H(x) $$

而且 $\ln$ 本质还可将乘法变成加法，这符合我们对熵表达

|  X | 0   | 1|
|---|---|---|
|  概率 | 1 - p | p|
|  信息量 | -ln (1 - p) | -ln p|
|期望|
大家都知道信息熵很大事件并不是大概率事件，所有我们根据期望公式对 logP 求期望就得
信息熵的概念
$$E(\log P) = -(1 - p) \cdot \ln(1 - p) - p \cdot \ln p$$
$$ E(\log P) = - \sum_{i=1}^n P_i \log P_i$$
上面离散情况下求信息熵
$$ H(X) \approx \int_0^{f(x)} \log f(x) dx $$

### 条件熵公式的推导

- 我们知道x 和 y 的熵，也知道 x 的熵
- 如果我们用 H(x,y) 减去 H(x) 的熵，那么就等价了 x 是确定性
- 也就是在已知 x 的条件下计算 y 的熵，这就是条件熵的概念
$$ H(X,Y) - H(X)$$

在 x 给定掉件 y 的信息熵，(x,y) 发生所包含的熵，减去 X 单独发生包含的熵:在 x 发生的前提下, y 发生**新**带来的熵。
$$ H(Y|X) = H(X,Y) - H(X) $$
下面我们来推导一下条件熵

$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} p(x) \log p(x)$$
式子加号右边中的 $p(x)$ 可以写出 $\sum_y p(x,y)$ ，这是因为 $ \sum_y p(x,y) $ 中 p(x,y) 对于 y 的积分，把所有的 y 加起来,进行积分去掉 y

$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} (\sum_y p(x,y)) \log p(x)$$

$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x,y} p(x,y) p(x) $$

$$ = - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)}$$
那么我们利用学过条件概率就知道$\frac{p(x,y)}{p(x)} = p(y|x)$

$$ = -\sum_{x,y} p(x,y) log p(y|x) $$

上面推导的公式结果$ -\sum_{x,y} p(x,y) log p(y|x) $ 要是能够变成$ -\sum_{x,y} p(y|x) log p(y|x) $ 形式我们看起来就会舒服多了。
###
$$ - \sum_x \sum_y p(x,y) \log p(y|x) $$
我们将$p(x,y)$ 替换为 $ p(y|x) p(x) $
$$ - \sum_x \sum_y p(x)p(y|x) \log p(y|x) $$
$$ - \sum_x p(x) \sum_y p(y|x) \log p(y|x) $$

$$ \sum_x p(x) ( - \sum_y p(y|x) \log p(y|x))$$
$- \sum_y p(y|x) \log p(y|x)$ 可以理解为 x 给定值时候 H(y) 因此我们就可以写成下面的公式
$$ = \sum_x p(x) H(Y|X = x)$$

### 相对熵
假设p(x),q(x) 是 X 中取值的两个概率分布，则 p 对 q 的相对熵是
$$ D(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = E_{p^{(x)}} = \log \frac{p(x)}{q(x)}$$

### 互信息
两个随机变量 X 和 Y 的互信息，定义为 X 和 Y 的联合分布和独立分布乘积的相对熵
$$ I(X,Y) = D(P(X,Y) || P(X)P(Y)) $$
$$ I(X,Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} $$

接下来我们看一看这个公式
$$ H(Y) - I(X,Y)$$
$$ = - \sum_y p(y) \log p(y) - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$
$$ - \sum_y \left( \sum_x p(x,y) \right) \log p(y) -   \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

$$ - \sum_{xy} p(x,y) \log p(y) - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} $$
$$ - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)}$$
$$ -\sum_{x,y} \log p(y|x) $$
$$ = H(Y|X) $$

$$ I(x,y) = H(Y) - H(Y|X) $$
#### 整理公式
条件熵定义
$$ H(Y|X) = H(X,Y) - H(X)$$
刚刚我们推导出公式如下
$$H(Y|X) = H(Y) - I(X,Y) $$
$$ I(X,Y) = H(Y) - H(Y|X) $$
根据对偶性
$$H(X|Y) = H(X,Y) - H(Y)$$
$$H(X|Y) = H(X) - I(X,Y) $$
$$I(X,Y) = H(X) + H(Y) - H(X,Y) $$
然后我们就能推导出
$$ H(X|Y) \le H(X)$$
$$ H(Y|X) \le H(Y)$$

![dt_12.jpeg](https://upload-images.jianshu.io/upload_images/8207483-b8225042ba6dd14a.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### KL 散度
$$ KL(p||q) = p \ln \frac{p}{q}$$
$$ = p(\ln p - \ln q)$$
$$ = p \ln p - p \ln q$$
$$ = - \left( - p \ln p + p \ln q   \right)$$
$$ = - \left( H(p) + p \ln q   \right)$$

- p 是实际样本分布
- q 是预测分布
- 希望我们预测分布尽量靠近实际分布
- 所以也就是期望 $KL(p||q)$ 越小越好，
### 如何找到策略
#### 离散情况
- 每一次找到作为判断特征都会找那个给我们信息的特征，也就是给我们互信息最多或是条件熵最小的特征
- 互信息在决策树中可以理解为信息增益

$$ Gain(D,a) = H(D) - \sum_{i=1}^m \frac{D_i}{D} H(D_i)$$
我们希望每一个信息增益是最大
接下来就来 ID3 算法，ID3 目标就是
- 什么条件可以判断决策树停止生长，也就是每个叶节点的纯度都
- Gini 系数 P(1-P)
$$ \begin{cases}
    p=1 & Gini = 0 \\
    p=0 & Gini = 0 
\end{cases} $$
- 可能出现过拟合。例如我们分的过细，剪掉分支
- 

$$ R = \sum P_k \log H_k(T) + \alpha|T| $$

#### 连续特征



### 决策树中算法
- ID3 信息增益
- C4.5 信息增益率
- CART Gini 系数

### 决策树的优点和缺点


--------------

## 决策树和随机深林

### CART(classification And Regression Tree)
通过字面可以理解为分类和回归树。









#### 信息熵

度量数据的纯度的指标这里用**信息熵**。一个事件发生概率与这个事件信息的关联，我们知道了解事件的信息越多那么这个事件发生概率，完全确定的也就是一定会发生，那么信息就会为 0
假设 x 和 y 是独立的事件，可以那么 x 和 y 两个事件的联合概率，
$$ P(x,y) = P(x)P(y)$$
我们希望信息量进行加减，如果将乘积变成去和的方式就会想到 log
$$ \log P(x,y) = \log P(x)P(y) = \log P(x) + \log P(y) $$

$H(x) H(y)$ 表示发生不确定性

$$ P(x,y) = $$

例如说太阳从东方升起那么这句话就是信息量$H(x)$为 0 的话，这里信息量是指事件本身，也就是小概率发生了那么他信息量就是很大。那么也就是说概率越大其携带的信息量就越少。

![分类](https://upload-images.jianshu.io/upload_images/8207483-11abca1629b0a5cd.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ A \{x_{apple},x_{banana},x_{strawberry},x_{banana},x_{orange} \dots x_{grape}, x_{orange} \} $$
$$ B \{x_{apple},x_{banana},x_{apple},x_{banana} \dots ,x_{banana} ,x_{banana}  \} $$

不难看出 A 集合和 B 集合，A 集合中种类中水果品种类别出现比较多，说明 A 集合比 B 集合的信息熵大。也就是所有 A 集合类别比较高。

![dt_03.jpeg](https://upload-images.jianshu.io/upload_images/8207483-01f2d938e390e56d.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


ln P 是增函数，那么 -ln P 就是一个减函数。

评价进行分割后好坏我们是通过信息熵来评价
$$ H(P) = -\sum_{i=1}^n p_i \ln p_i $$
也可以用阻尼系数作为判断集合纯度的指标
$$Gini(p) = \sum_{k=1}^k p_k(1-p_K) = \sum_{k=1}^k p_k^2$$

构造树的基本思想是随着深度的增加，节点的熵迅速地下降，熵降低速度越快越好



### 条件熵



###
$$ - \sum_x \sum_y p(x,y) \log p(y|x) $$
$$ - \sum_x \sum_y p(x)p(y|x) \log p(y|x) $$
$$ - \sum_x p(x) \sum_y p(y|x) \log p(y|x) $$

$$ \sum p(x) ( - \sum_y p(y|x) \log p(y|x))$$
$$ = \sum_x p(x) H(Y|X = x)$$
### 优点
- 训练树训练快

|  X | R   | G|
|---|---|---|
|  N | 70 | 65|
|  P | 51.8% | 48.2%|

###
- ID3 信息增益
- C4.5 信息增益率
- CART Gini 系数

### 目标函数
$$ C(T) = \sum_{releaf} N_t \cdot H(t) $$


###
$$ \sum_{x,y}p(y|x) \log p(y|x) $$