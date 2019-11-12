
![dt_01.jpg](https://upload-images.jianshu.io/upload_images/8207483-29675bd966b05172.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 决策树(Decisino Tree)
决策树(Decision Tree）是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。
### 前言
我们将讨论可以用于**分类**和**回归**任务的简单、非线性的模型——**决策树**。个人认为决策树相对于其他机器学习模型更容易理解，符合我们人类思考和对事物进行分类判断的逻辑。我们在现实生活中就是这样一步一步做出决策的。例如找对象、找工作都会不自觉用到决策树算法来做出决定我们的决定。有人可能认为现在是深度学习的天下，决策树是不是已经过时了没有立足之地了。虽然在某些应用已经打上深度神经网的标签，但是在一些领域中可能更适合选择决策树而非深度神经网模型。
### 决策树
![dt_011.jpeg](https://upload-images.jianshu.io/upload_images/8207483-99aa3dc5a7bcc4d9.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

决策树是一个**非线性**分类器,给出以一种分割数据的规则来将数据切分为 $N_1$ 和 $N_2$ 两份数据。既然这里提到了线性和非线性模型，我们简单介绍一下什么是线性模型和什么是非线性模型，以及两者的区别。
![dt_03.jpg](https://upload-images.jianshu.io/upload_images/8207483-353e05689bd393ee.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 线性模型
在统计意义上，如果一个回归等式是线性的，其参数就必须也是线性的。如果相对于参数是线性，那么即使性对于样本变量的特征是二次方或者多次方，这个回归模型也是线性的。
$$ y = \theta_0 + \theta_1x_1 + \theta_2x_2^2$$
#### 非线性模型
最简单的判断一个模型是不是非线性，就是关注非线性本身，判断其参数是不是非线性的。
$$ y = \theta_0 + \theta_1 x_1 + \theta_2^2 x_2 $$

- $x_1$ 和 $x_2$ 不是独立，在不同区域内 $x_1$ 和 $x_2$ 关系不同。

#### 线性模型和非线性模型的区别
在机器学习的回归问题中，**线性模型**和**非线性模型**都可以去对曲线进行建模，所有有许多人误认为线性模型无法对曲线进行建模。其实，线性模型和非线性模型的区别并不在于能不能去拟合曲线。
1. 线性模型可以用曲线拟合样本，但是分类的决策边界一定是直线(如 logistics 模型)
2. 区分是否为线性模型，主要是看一个乘法式子中自变量 $x$ 前的系数 $\theta$ ，应该是说 $x$ 只被一个 $\theta$ 影响，那么此模型为线性模型。
3. 画出 $y$ 和 $x$ 是曲线关系，但是他们是线性模型，因为 $theta_1x_1$ 中可以观察到 $x_1$ 只被一个 $w_1$ 影响
   $$ y = \frac{1}{e^{(\theta_0,+ \theta_1x_1 + \theta_2x_2)}}$$
4. 当(如下面模型中) $x_1$ 不仅仅被参数$\theta_1$ 影响，而且还被$\theta_5$ 影响，自变量 $x$ 被两个以上的参数影响，那么模型就是非线性模型
    $$ y = \frac{1}{1 + \theta_5 e^{(\theta_0,+ \theta_1x_1 + \theta_2x_2)}}$$

![dt_08.jpg](https://upload-images.jianshu.io/upload_images/8207483-f155693ba75b931f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 看一个例子
1. 看待遇
2. 看是否双休
3. 看是否出差
4. 看是否
5. 看是否有发展
通过系列问题我们衡量最终结果是去还是不去这家公司。这系列问题就是



#### 决策树的结构
![dt_09.jpg](https://upload-images.jianshu.io/upload_images/8207483-17e85d2b41255fd9.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

决策树算法是用树状结构表示数据分类的结果
- 根节点(root node)
- 非叶子节点(internal node)
- 叶子节点(leaf node)每一个叶子节点都有一个分类值
- 分支(branch)
![dt_07.jpg](https://upload-images.jianshu.io/upload_images/8207483-4b80d252e374b616.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在开始之前，我们看一些重要概念以及公式推导。
### 信息熵
熵概念最早是在物理热力学中提出概念后来也被在统计学中使用，表示提出了**信息熵**的概念，用以度量信息的不确定性。度量数据的纯度的指标这里用**信息熵**。一个事件发生概率与这个事件信息的关联，完全确定的也就是一定会发生，那么信息就会为 0。既然我们知道信息熵和概率的关系我们下面来推导一下**信息熵**计算公式
$H(x) H(y)$ 表示事件发生不确定性，也就是事件信息熵，他们分别表示x 和 y 的信息熵。

例如说太阳从东方升起那么这句话就是信息量$H(x)$为 0 的话，这里信息量是指事件本身，也就是小概率发生了那么他信息量就是很大。那么也就是说概率越大其携带的信息量就越少。

![dt_10.jpg](https://upload-images.jianshu.io/upload_images/8207483-10c1a5c609d2357e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们通过下面实际例子来说说信息熵，下面有 A 和 B 两个集合分别是不同类别的元素，在 A集合包含各种不同种类水果的元素，而相对于 A 集合 B 集合包含了属于较少类别的元素。


![分类](https://upload-images.jianshu.io/upload_images/8207483-11abca1629b0a5cd.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ A \{x_{apple},x_{banana},x_{strawberry},x_{banana},x_{orange} \dots x_{grape}, x_{orange} \} $$
在 A 集合中有属于 banana apple straberry orange grape 等不同类别的元素
$$ B \{x_{apple},x_{banana},x_{apple},x_{banana} \dots ,x_{banana} ,x_{banana}  \} $$
而在 B 集合中仅包含较少类别的水果

不难看出 A 集合和 B 集合，A 集合中种类中水果品种类别出现比较多，说明 A 集合比 B 集合的信息熵大。也就是所有 A 集合类别比较高。
下面图为$ -ln(x)$
![屏幕快照 2019-11-10 下午4.56.16.png](https://upload-images.jianshu.io/upload_images/8207483-73032c11e7f169dc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


ln(x) 是增函数，那么 -ln(x) 就是一个减函数。所有在 x = 1 处 -ln(x) 为 0 而在 x = 0 时候 -ln(x) 却是无穷大

下面公式就是信息熵的公式
- 在离散情况下信息熵公式
$$ H(x) = -\sum_{i=1}^n P(x_i) \log P(x_i) $$

- 在连续函数情况下信息熵的公式
$$ H(P) = - \int p(x) lnp(x) dx$$

#### 联合概率
假设 x 和 y 是独立的事件，可以那么 x 和 y 两个事件的联合概率，
$$ P(x,y) = P(x)P(y)$$
我们希望信息量进行加减而不是乘法，如果将乘积变成去和的方式就会想到 log 来解决对于公式两边都取 log
$$ \log P(x,y) = \log P(x)P(y) = \log P(x) + \log P(y) $$


### 条件熵
用于在有信息情况的信息，
在 x 给定掉件 y 的信息熵，(x,y) 发生所包含的熵，减去 X 单独发生包含的熵:在 x 发生的前提下, y 发生**新**带来的熵。也就是如果已知 x 的条件下是否给 y 的熵带来变化。


$$ H(y|x) = H(X,Y) - H(X) $$???
首先我们看一下熵的一些基本操作
- 在 x 和 y 都是相互独立
$$ H(x,y) = H(x) + H(y)$$
- 在 x 和 y 并非是相互独立
$$ H(x,y) < H(x) + H(y)$$

我们这里对 $$H(x,y)$$ 进行简单推导
根据信息熵的概念我们可以推导下面公式
$$ H(x,y)=  - \sum_{x,y} P(x,y)ln(P(x,y)) $$
然后根据$P(x,y) = p(y|x)P(x)$ 带入下面等式就可以得到下面公式
$$ = -\sum_{x,y} P(x,y) \ln (P(x)P(y|x)) $$
进一步推导为
$$ = - \sum_{x,y}  P(x,y)(\ln P(x) + \ln P(y|x)) $$
$$ = - \sum_{x,y}  P(x,y) \ln P(x) - \sum_{x,y} P(x,y) \ln P(y|x)) $$
因为这里我们只对 x 进行积分(求和)所有可以在前半部化简为
$$ = - \sum_{x}  P(x) \ln P(x) - \sum_{x,y} P(x,y) \ln P(y|x)) \tag{1}$$
$$ = H(x) - \sum_{x,y} P(x,y) \ln P(y|x)) $$
<!-- 到现在为止我们整理一下思路，我们???
$$ \begin{cases}
    H(y|x) = H(x,y) - H(x) \\
    I(x,y) = H(x,y) - H(x) - H(y) \tag{1}
\end{cases} $$ -->
### 互信息
**互信息**指的是两个随机变量之间的关联程度，即给定一个随机变量后，另一个随机变量不确定性的削弱程度。只有真正理解好了**互信息**我们才能够理解在决策树背后是如何通过节点来降低数据的信息熵从而达到分类和回归的效果。

互信息，也就是度量信息，其实互信息就是度量两个概率，那么也就是独立情况下互信息为 0，如果不独立就是大于 0 的某个数，这个数表示两个概率距离。

$$ I(x,y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} $$
![dt_11.jpeg](https://upload-images.jianshu.io/upload_images/8207483-cfd1ceffe6b35d02.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$I(x,y) = H(x) - H(x|y) $$
$$I(x,y) = H(y) - H(x|y) $$
$$I(x,y) = H(x) + H(y) - H(x,y)$$

这是因为我们看一下下面推导就可以明白为什么$I(x,y)$ 可以作为度量两个不相互独立之间熵的影响，
$$
    \begin{aligned}
        H(x,y) = H(x)+ H(y) \\
        H(x,y) < H(x) + H(y) 
    \end{aligned}
$$

$$ H(y|x) = H(x,y) - H(x) = -\sum_{x,y} P(x,y) \log P(y|x) $$
等效证明互信息是大于等于 0，今天证明的确有点烧脑，不过对我们能够理解决策树中一些算法还是有帮助的。

$$ I(x,y) = H(x,y) - H(x) - H(y) $$
只要证明了$ I(x,y) \le 0 $ 也就是说明$H(x,y) \le H(x) + H(y)$,也就证明了 
$$ H(y|x) - H(y) = - (\sum_{x,y} \log P(y|x)) + \sum_{x,y} P(x,y) \log P(y) $$


$$ = -\left[ \sum_{x,y} P(x,y)(\log \frac{P(x,y)}{P(x)} - \log P(y)) \right]$$
$$ = -\sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} $$

### 构建决策树
我们有了上面的基础知识，了解了什么是条件熵和互信息，表示我们可以通过给定特征来降低信息熵，我们回到决策树的问题上来看一看。
我们如何设计一颗决策树，选择那给个条件作为根节点，有了上面知识，我们不难使用公式来判断可以快速降低数据的信息熵的节点作为首要节点。
![dt_016.jpg](https://upload-images.jianshu.io/upload_images/8207483-ae3e17f56ead485a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 从 ID3 到 C4.5
- ID3 定义
**ID3 算法**的核心是在决策树各个子节点上应用信息增益准则选择特征，递归的构建决策树，具体方法是:从根节点开始，对节点计算所有可能的特征的**信息增益**，选择信息增益最大的特征作为节点的特征，由该特征的不同取值建立子节点；再对子节点递归调用以上方法，构建决策树。

我们通过一个具体实例来讲解如何通过 ID3 算法来选择节点。

#### 求职问题

![dt_015.jpg](https://upload-images.jianshu.io/upload_images/8207483-fa44bbf400979803.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

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
先复习一下什么是信息熵
- 熵概念: 信息熵是用来描述信息的混乱程度或者信息的不确定度。信息熵是可以反映集合的纯度

我们要计算纯度也就是计算其不确定度，不确定度大其纯度就低。例如 A 和 B 两个事件的概率都是 30% 50%，A 事件的不确定性(也就是信息熵) 要大 。算法目的通过不断决策树来将信息熵降低下来
我们先将公式推出。

$$ Ent(D) = \sum_{k=1}^m p_k \log_2 \frac{1}{p_k} = - \sum_{k=1}^m p_k \log_2 p_k$$


- 信息增益 $$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

- 信息增益率
$$ Gain(D,C) = Ent(D) - Ent(D|C) = Ent(D) - \sum_{i=1}^n \frac{N(D_i)}{N} Ent(D_i) $$

#### 分析实例
我们以招聘为例，从样本看接受 offer 和没有接受 offer 各占 50% 所以他们概率分别是

$$  \begin{cases}
    P_{recive} = \frac{1}{2} \\
    P_{refuse} = \frac{1}{2}
\end{cases} $$
$$ - ( \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2}) = 1$$

```
-(0.5 * math.log(0.5,2) + 0.5 * math.log(0.5,2))
```

#### 划分条件为待遇情况
我们希望分支之后信息熵是降低的，我么选择条件 C 后这两个节点，我们回去查表看看高薪岗位的有几个，一共有 6 个岗位。那么是高薪招聘到人是 5 

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
表示划分后信息熵为，信息增益越大越好，使用信息熵天生对多路分类进行偏好。假设我们有 id 作为条件,那么每一个 id 得到结果是确定也就是概率是 1 那么 $$- 1 * \ln 1 = 0 $$
$$ \sum_{i=1}^n (-ln1) = 0 $$
那么他信息增益就是最大，但是我们先验不会选择 id 作为
- 信息增益率
$$ Gain_ratio(D,C) = \frac{Gain(D,C)}{Ent(C)} $$

$$ Ent(C) = - \sum_{i=1}^k \frac{N(D_i)}{N} \log_2 \frac{N(D_i)}{N} $$

对于多路划分乘以代价
$$ Gain(D C_1) = 0.61, Gain(D C_2) = 0.72$$
$$ Ent(C_1) = - (\frac{6}{10} \log \frac{6}{10} + \frac{4}{10} \log \frac{4}{10}) = = 0.971$$
$$ Ent(C_2) = 1.571$$

$$ Gain_{ratio} = 0.63  $$

$$ $$

##### 延森不等式
![jensen-small.png](https://upload-images.jianshu.io/upload_images/8207483-300ae6cdca2368a8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
$$ \alpha_1 \log x_1 + \alpha_2 \log x_2 \le \log (\alpha_1x_1 + \alpha_2x_2)$$


再用其他方式下面我们来推导一下条件熵

$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} p(x) \log p(x)$$
有了这个不等式
$$ -\sum_{x,y}P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$
$$ =  \sum_{x,y}P(x,y) \log \frac{P(x)P(y)}{P(x,y)} $$
$$ =  \sum_{x,y} \log P(x,y) \frac{P(x)P(y)}{P(x,y)} $$
$$ \ge  \sum_{x,y} \log P(x)P(y) $$
上面不等式可以看出联合概率熵和要大于各种概率熵的和
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
- 也称为相对熵、鉴别信息 Kullback 熵，Kullback-Leible 散度等
- 假设 p(x) q(x) 是 X 中取值的两个概率分布，则 p 对 q 的相对熵就是
$$ D(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = E_{p(x)} \log \frac{p(x)}{q(x)} $$
- 说明
    - 交叉熵可以度量两个随机变量的**距离**
    - 一般 $ D(p||q) D(q||p) $
    - $ D(p||q) \ge 0 D(q||p) \ge 0 $
$$ p(x) \approx q(x) $$
什么是交叉熵，P(x) 是未知的，我们实际上用q(x) 来逼近 p(x)
我们只能根据数据推测 P(x)

$$ H_{cross} = - \int P(x) \log q(x) dx $$
$$ H_{cross} \ge S = - \int p(x) \log P(x) dx $$

接下我们来一起证明一下
$$ H(q(x)) - H(p(x))$$
$$ = - \int p(x) \log q(x) dx + \int p(x) \log p(x) dx $$
$$ = - \int p(x) \log \frac{q(x)}{p(x)} dx \approx KL $$
$$ H(q(x)) - H(p(x))  $$
$$ -\log \int q(x) dx$$

我们从另一个角度进行推导
把所有的 y 加起来进行积分后积分掉 x,
$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} (\sum_y p(x,y)) \log p(x)$$
$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x,y} p(x,y) \log p(x)$$

$$ - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x,y} p(x,y) p(x) $$

$$ - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)}$$
$$ -\sum_{x,y} p(x,y) log p(y|x) $$

## 公式推导

----

### 信息熵公式的推导
学过概率后,如果两个事件 x 和 y 是独立，这个应该不难理解$$ P(x,y) = P(x) \cdot P(y)$$ 
首先我们已经知道了信息熵和概率关系是成反比，也就是说概率越大信息熵越小，这个应该也不难理解，如果一个一定会发生事，例如太阳从东方升起，这个事件给我们带来信息量非常小
![dt_05.jpeg](https://upload-images.jianshu.io/upload_images/8207483-486e514a2beeaf77.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们知道$\ln x$函数模样如下图，$\ln x$ 是一个增函数
![lnx](https://upload-images.jianshu.io/upload_images/8207483-5167cb7e309e678b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果对函数添加负号就变成减函数，而且在 0 - 1 区间函数值符号我们要求就是在 1 时函数值为 0 而接近 0 时为无穷大
![-lnx](https://upload-images.jianshu.io/upload_images/8207483-cfd0f1caf3c84759.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然后我们进步想象一下如果希望两个事件的信息量也是可以做加减
$$H(y|x) = H(x,y) - H(x) $$

而且 $\ln$ 本质还可将乘法变成加法，这符合我们对熵表达

|  X | 0   | 1|
|---|---|---|
|  概率 | 1 - p | p|
|  信息量 | -ln (1 - p) | -ln p|
|期望|
大家都知道信息熵很大事件并不是大概率事件，所有我们根据期望公式对 logP 求期望就得
信息熵的概念
$$E(\log P) = -(1 - p) \cdot \ln(1 - p) - p \cdot \ln p$$
$$ E(\log P) = - \sum_{i=1}^n P_i \log P_i$$
上面离散情况下求信息熵
$$ H(X) \approx \int_0^{f(x)} \log f(x) dx $$

### 条件熵公式的推导

- 我们知道x 和 y 的熵，也知道 x 的熵
- 如果我们用 H(x,y) 减去 H(x) 的熵，那么就等价了 x 是确定性
- 也就是在已知 x 的条件下计算 y 的熵，这就是条件熵的概念
$$ H(X,Y) - H(X)$$

在 x 给定掉件 y 的信息熵，(x,y) 发生所包含的熵，减去 X 单独发生包含的熵:在 x 发生的前提下, y 发生**新**带来的熵。
$$ H(Y|X) = H(X,Y) - H(X) $$
下面我们来推导一下条件熵

$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} p(x) \log p(x)$$
公式中$p(x)$可以写出$\sum_y p(x,y)$ 把所有的 y 加起来,进行积分去掉 y
$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} (\sum_y p(x,y)) \log p(x)$$
$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x,y} p(x,y) \log p(x)$$

$$ = - \sum_{x,y} p(x,y) \log p(x,y) + \sum_{x,y} p(x,y) p(x) $$

$$ = - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)}$$
那么我们利用学过概率就知道$\frac{p(x,y)}{p(x)} = p(y|x)$

$$ = -\sum_{x,y} p(x,y) log p(y|x) $$

上面推导的公式结果$ -\sum_{x,y} p(x,y) log p(y|x) $ 要是能够变成$ -\sum_{x,y} p(y|x) log p(y|x) $ 形式我们看起来就会舒服多了。
###
$$ - \sum_x \sum_y p(x,y) \log p(y|x) $$
我们将$p(x,y)$ 替换为 $ p(y|x) p(x) $
$$ - \sum_x \sum_y p(x)p(y|x) \log p(y|x) $$
$$ - \sum_x p(x) \sum_y p(y|x) \log p(y|x) $$

$$ \sum_x p(x) ( - \sum_y p(y|x) \log p(y|x))$$
$- \sum_y p(y|x) \log p(y|x)$ 可以理解为 x 给定值时候 H(y) 因此我们就可以写成下面的公式
$$ = \sum_x p(x) H(Y|X = x)$$

### 相对熵
假设p(x),q(x) 是 X 中取值的两个概率分布，则 p 对 q 的相对熵是
$$ D(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = E_{p^{(x)}} = \log \frac{p(x)}{q(x)}$$

### 互信息
两个随机变量 X 和 Y 的互信息，定义为 X 和 Y 的联合分布和独立分布乘积的相对熵
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
$$ I(x,y) = H(Y) - H(Y|X) $$
根据对偶性
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
- 所以也就是期望 $KL(p||q)$ 越小越好，
### 如何找到策略
#### 离散情况
- 每一次找到作为判断特征都会找那个给我们信息的特征，也就是给我们互信息最多或是条件熵最小的特征
- 互信息在决策树中可以理解为信息增益

$$ Gain(D,a) = H(D) - \sum_{i=1}^m \frac{D_i}{D} H(D_i)$$
我们希望每一个信息增益是最大
接下来就来 ID3 算法，ID3 目标就是
- 什么条件可以判断决策树停止生长，也就是每个叶节点的纯度都
- Gini 系数 P(1-P)
$$ \begin{cases}
    p=1 & Gini = 0 \\
    p=0 & Gini = 0 
\end{cases} $$
- 可能出现过拟合。例如我们分的过细，剪掉分支
- 

$$ R = \sum P_k \log H_k(T) + \alpha|T| $$

#### 连续特征



### 决策树中算法
- ID3 信息增益
- C4.5 信息增益率
- CART Gini 系数

### 决策树的优点和缺点


--------------

## 决策树和随机深林

### CART(classification And Regression Tree)
通过字面可以理解为分类和回归树。









#### 信息熵

度量数据的纯度的指标这里用**信息熵**。一个事件发生概率与这个事件信息的关联，我们知道了解事件的信息越多那么这个事件发生概率，完全确定的也就是一定会发生，那么信息就会为 0
假设 x 和 y 是独立的事件，可以那么 x 和 y 两个事件的联合概率，
$$ P(x,y) = P(x)P(y)$$
我们希望信息量进行加减，如果将乘积变成去和的方式就会想到 log
$$ \log P(x,y) = \log P(x)P(y) = \log P(x) + \log P(y) $$

$H(x) H(y)$ 表示发生不确定性

$$ P(x,y) = $$

例如说太阳从东方升起那么这句话就是信息量$H(x)$为 0 的话，这里信息量是指事件本身，也就是小概率发生了那么他信息量就是很大。那么也就是说概率越大其携带的信息量就越少。

![分类](https://upload-images.jianshu.io/upload_images/8207483-11abca1629b0a5cd.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ A \{x_{apple},x_{banana},x_{strawberry},x_{banana},x_{orange} \dots x_{grape}, x_{orange} \} $$
$$ B \{x_{apple},x_{banana},x_{apple},x_{banana} \dots ,x_{banana} ,x_{banana}  \} $$

不难看出 A 集合和 B 集合，A 集合中种类中水果品种类别出现比较多，说明 A 集合比 B 集合的信息熵大。也就是所有 A 集合类别比较高。

![dt_03.jpeg](https://upload-images.jianshu.io/upload_images/8207483-01f2d938e390e56d.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


ln P 是增函数，那么 -ln P 就是一个减函数。

评价进行分割后好坏我们是通过信息熵来评价
$$ H(P) = -\sum_{i=1}^n p_i \ln p_i $$
也可以用阻尼系数作为判断集合纯度的指标
$$Gini(p) = \sum_{k=1}^k p_k(1-p_K) = \sum_{k=1}^k p_k^2$$

构造树的基本思想是随着深度的增加，节点的熵迅速地下降，熵降低速度越快越好



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


