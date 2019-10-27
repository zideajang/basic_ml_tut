
## 前言
当下机器学习比较重要 3 中算法，个人都目前为止认为比较重要机器学习算法分别是，深度学习、SVM 和决策树。在深度学习出现之前，是 SVM 的时代 SVM 占据了机器学习算法优势地位整整 15 年。
## SVM VS 深度学习
深度学习是对原始特征重新表示，也就是原始特征虽然学的很懒，通过。就拿深度学习擅长的图像识别，是因为我们人类使用 pixel 来表示图片，用 pixel 来表示图片是一种比较烂表示方式，这是通过深度学习能够帮助我提取一些新表示特片特征来对图像进行表示。
而且深度学习有时候即使对于设计者是个黑核模式，也就是我们不知道机器如何学习来给出其预测或推荐。有时候我们是需要知道机器是如何学习，学习到了什么，以及如何做出决定和推荐。

而 SVM 好处是通过数学推导出来的模型。个人尝试了解一下感觉还是比较难。
## SVM VS 决策树
SVM 和决策树都是找决策边界

## SVM(Support Vector Machine)

SVM = Hinge Loss + (核方法)Kernel Method

### 概念 SVM (Support Vector Machine)
SVM 是我们必须掌握统计算法，而且 SVM 推导过程要比其应用更加重要。使用 SVM 前提是我们分类问题是线性可分的。

SVM 用于解决分问题的分类器。介绍且背后算法，了解算法后便于我们更好调整参数。在深度学习出现之前被大家广泛应用用于分类算法。
![svm](https://upload-images.jianshu.io/upload_images/8207483-4d939008a9e07a26.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
图中有我们用颜色来表示不同样本，所谓分类问题就是找到一条线，让线两边分别是不同的类别。$L_1$ 

我么不仅需要关注训练误差，我们更要关系期望损失
### SVM 关键术语
- 间隔: 就是离
- 对偶
- 核技巧:在 SVM 出现之前就有核技巧，通过核技巧让 SVM 从欧式空间扩展到多维空间
### SVM 分类
- 硬间隔 SVM
- 软间隔 SVM
- 核 SVM

### SVM 推导过程
- 监督学习
我们对比深度学习 SVM，深度学习通过梯度下降不断优化边界，然后最后找到最优解而 SVM 无需梯度下降就可以
- 几何含义
就是我们要找的决策分界线到离其最近点的边距最大。这就是 SVM 的几何含义。那么有了这个想法我们就需要用数学来描述我们的想法，这也就是找模型的过程。
- 决策
![support-vector-machine-and-implementation-using-weka-19-638.jpg](https://upload-images.jianshu.io/upload_images/8207483-98fbf8be1a4a64e2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ $$
### Hinge Loss
1. 找到函数模型集合
这里我们依旧是进行二分类问题，我们在训练集样本 $\{ x^1 x^2 \cdots x^i \}$ 对于标签为 $\{ \hat{y}^1 \hat{y}^2 \cdots \hat{y}^n \}$ 这里用-1 和 1 表示结果。
$$ g(x)=\begin{cases}
f(x) > 0,Output = 1 \\ 
f(x) < 0,Output = -1
\end{cases} $$
定义 g(x) 函数，其中包含一个函数 f(x) 当 f(x) > 0 g(x) 为 1 反之为 -1。
2. **损失函数**
$$ L(f) = \sum_n \delta(g(x^n) \neq \hat{y}^n ) $$
对于所有数据 $\delta$ 表示内部表达式为真则为 1 否则为 0。那么当 g(x) 不等于 $\hat{y}^n$ 时候输出为 1 ，损失函数计算函数就记录一次，求和后会记录犯了几次错误。但是很明显这个函数无没有办法微分的。
$$ L(f) = \sum_n \hat{y}^n f(x)$$
![svm_1](https://upload-images.jianshu.io/upload_images/8207483-010bfd7f38821f2f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- Square Loss
$$ if  \hat{y}^n = 1 , f(x) close to 1 $$
$$ if  \hat{y}^n = -1 , f(x) close to -1 $$
$$ l(f(x^n),\hat{y}^n) = (\hat{y}^nf(x^n) - 1)^2$$
然后我们把 $\hat{y} = 1 $ 带入方程得到 $(f(x^n) - 1)^2$，也就是说明当  $\hat{y} = 1 $  时候 $f(x^n)$ 越接近 1 也好。
然后我们把 $\hat{y} = -1 $ 带入方程得到 $(f(x^n) + 1)^2$，$\hat{y} = -1 $  时候 $f(x^n)$ 越接近 -1 也好。
- Sigmoid + Square Loss
$$ if  \hat{y}^n = 1 , f(x) close to 1 $$
$$ if  \hat{y}^n = -1 , f(x) close to -1 $$
$$ l(f(x^n),\hat{y}^n) = (\hat{y}^nf(x^n) - 1)^2$$




#### 参数分析
- c 用来调整松弛变量系数
- kernel 核函数用进行判别类别的函数 例如 rbf
- degree
- gamma 是关于 rbf 的特有参数
- probability 学习统计同学比较了解，结论的可能性，通过统计理论以及
- cache_size 对 SVM 计算缓存大小
- class_weight=None 每一个类别比较平均就不需要设置这个参数，样本不均匀时候是不会用到。

### SVM 分类
#### 线性
#### rbf 

### 什么是 SVM 算法
- 二元线性分类问题(简单)
    - 可分问题
    - 什么样线性方程是最好线性的方程，离这条子线最近那些点离这条线最远，这也是 SVM 的目标
    - 有很多判别线
    - 支持向量与我们直线最近那些点(向量)就是支持向量

- 回忆解析几何，点到直线的距离
- $$ (x,y) $$

$$  f(x) = W^Tx + b $$
$$ V^i = \frac{W^T}{||W||} x^i + \frac{b}{||W||}$$
$$ V^i = \frac{W^T}{||W||} x^i + \frac{b}{||W||}$$

### 线性 SVM
#### 找出方程组
$$ f(x) = \sum_i w_ix_i + b = \left[ \begin{matrix}
    w \\ 
    b
\end{matrix} \right] \cdot \left[ \begin{matrix}
    x \\ 
    1
\end{matrix} \right] = W^T x$$

#### 损失函数

#### 梯度下降