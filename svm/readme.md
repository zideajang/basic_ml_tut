### 为什么要学习 SVM
今天深度学习算法大大提高准确度，固然深度学习能够交给我们一张满意答卷。由于深度学习通过大量仿生神经元的设计来完成暴力方式进行学习，所以我们无法向客户解释机器是如何通过学习来解决回归和分类的问题。特别在医疗领域上，我们需要向客户解释机器是如何根据患者的报告来对患者病情进行诊断。因为深度神经
大多情况下对于开发者是黑盒，我们无法了解机器是如何一步一步根据。这样有关人命关天的大事，客户是无法接受机器做出诊断。

![svm_1.jpeg](https://upload-images.jianshu.io/upload_images/8207483-07eef9256851521c.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
那么什么是 SVM 呢? SVM 是 Support Vector Machine 英文缩写，直接翻译过来就是支持向量机。SVM 是一种机器学习算法，用于处理二分类和多分类问题。SVM 是监督学习的一种。


![svm](https://upload-images.jianshu.io/upload_images/8207483-69ef528777c610d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

SVM 、决策树和深度学习，是当下机器学习世界里 3 个比较重要的算法。早在深度学习都目前为止认为比较重要机器学习算法分别是，在深度学习出现之前，SVM 和决策树占据了机器学习。那时候是 SVM 的时代 SVM 占据了机器学习算法优势地位整整 15 年。


我们在开始学习SVM 之前需要对一些 SVM 算法用到数学知识进行简单回顾一下。

#### 介绍分割边界（超平面)
这回在坐标系(特征空间)中绘制一条直线，该直线在 x 轴和 y 轴上的截距分别是 -2 和 1。我们根据这些已知条件，利用初中的数学就可以很轻松写出这条直线所对应的方程。 
![svm_02](https://upload-images.jianshu.io/upload_images/8207483-0649daadfff67562.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ y = \frac{1}{2} x + 1$$
对等式进行简单的化简就可以得到
$$ -x + 2y + 2 = 0  $$
$$ f(x,y) = -x + 2y + 2 = 0  $$
我们知道将函数等于 0  这个函数就变为一个方程，也可以使用向量的方式表示这个方程

$$ f(x,y) = \left[ \begin{matrix}
    -1,2
\end{matrix} \right]  \left[ \begin{matrix}
    x \\
    y
\end{matrix} \right] + 2$$
系数 w 就是方程的法向方向，可以表示为 $\vec{w} = [-1,2]^T$

w 右上角的 T 表示对向量的转置也就是行列对调，这个之前我们已经遇到过，所以上面方程就可以写成下面的式子。

$$ \vec{w}^T \cdot \vec{x} + b $$
不少书籍上就直接写成$wx + b = 0$

位于该直线法向方向的所有点的集合,带入到 wx + b 就满足下面不等式，我们通常把这些点叫正例样本

$$ \vec{w}^T \cdot \vec{x} + b > 0 $$ 

位于法线反方向上的点满足下面不等式，这些点就是负例样本。

$$ \vec{w}^T \cdot \vec{x} + b < 0 $$ 

表示点位于该直线上，点满足下面等式

$$ \vec{w}^T \cdot \vec{x} + b = 0 $$ 

现在对直线进行扩展，直线上有两个特征$x_1,x_2$ 表示一条直线，而如果有三个特征$ \{ x_1,x_2,x_3 \}$ 表示空间中一个平面，如果是 N 维的话就无法通过图像来表示，那么这就是我们所说在 N维空间上超平面。其实直线和平面也是超平面。这些应该不难理解。

好了我们知道如何通过一个超平面把数据样本点分类正例样本和负例样本两类。现在我们来思考一个问题就是如何判断集合点存在一个边界线。如何判断数据集是线性可?如果存在一个超平面，所有特征点在这个超平面法线方向上的投影是可分的，那么这个数据集就是**线性可分**的。

![svm_02.jpeg](https://upload-images.jianshu.io/upload_images/8207483-72fab6464568d1a4.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在左图中我们看到在两类样本点存在无穷多个分割平面(边界平面)可以把两类样本点分离开。现在问题是我们如何选择一个最佳分割平面呢?图中有些一些线我们根据经验一看就会发现。
![svm_03](https://upload-images.jianshu.io/upload_images/8207483-0896cf2e2195e017.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

L2 分割平面比较贴近红色方块的点，而 L2 更贴近蓝色圆圈点。这两边界都是有缺点，显然这些分割线没有良好泛化能力，我们凭借经验会发现好的分割平面应该是在中间地带，离两边样本点都尽量大。

![SVM_06](https://upload-images.jianshu.io/upload_images/8207483-946c1edc5fae6a71.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其实根据我们经验我们是大概知道什么样分割平面是最好，那么我们需要严谨地用数学语言进行描述一下什么我们想要决策平面。

这个两面需要离两个数据集合点距离最大，并且分割平面到离他最近点的距离最大。在机器学习中很多问题都是最大最小的问题。这里离分割平面最近点叫作为支持向量点，他们是机器学习关键点
![svm_07.jpg](https://upload-images.jianshu.io/upload_images/8207483-3ecd4310cb4b2530.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 点到直线距离
这里我们来看一看点$(x_0,y_0)$到$ Ax + By + C = 0$直线的距离，点到直线距离公式是不是初中的知识点已经不记得了。

$$ f(x_0,y_0) = \frac{|Ax_0 + By_0 + C|}{\sqrt{A^2 + B^2}} $$
对公式进行化简，
$$ \frac{A}{\sqrt{A^2 + B^2}} x_0 + \frac{B}{\sqrt{A^2 + B^2}} y_0 + \frac{C}{\sqrt{A^2 + B^2}}$$
其实点到直线方程就可以写出这个样子
$$ A\prime x_0 + B \prime y_0 + C \prime  $$
利用我们在前面学到直线表达式，就会得到
$$(A\prime  + B \prime ) \cdot \left( x_0,y_0 \right)^T+ C$$
也就是 $\vec{w}^T \vec{x} + b$，不也就是一条直线那么，也就是将点到直线的距离直线，扩展到 n 维，w 为直线法线的方向然后我们对 
$$ (w_1,w_2, \dots w_n) $$
向量的模如下
$$ ||w||_2 = (w_1,w_2, \dots w_n) \cdot (w_1,w_2, \dots w_n)^T$$

$$ N = \{(x_1,y_1),(x_1,y_1),\dots ,(x_n,y_n) \} $$
$$ x_i \in \mathbb{R} $$



$$f(x,w,b) = 0$$ 表示直线，假设已知 w 和 b 那么我们就得到一条直线
- 对于正例样本，对于所有正例样本其标签为 $y^{possive}$ 为 1，所以整数距离乘以 1 为整数
$$ \frac{w^{possive} x + b }{||w||} y^{possive} $$
- 对于负例样本，对于所有正例样本其标签为 $y^{negtive}$ 为 -1，所以负整数距离乘以 -1 为正整数。
$$ \frac{w^{negtive} x + b }{||w||} y^{negtive} $$

$$ \frac{w x^{i} + b }{||w||} y^{i} $$
这是所有样本到分割平面距离，我们接下来就要找距离分割平面最近点
$$ \min_{i=1,2 \dots n} \frac{w x^{i} + b }{||w||} y^{i} $$
这是表示所有样本点$（x_i,y_i）$到某一条直线$f(w_j,b_j)$的最近点，

$$ max( \min_{i=1,2 \dots n} \frac{w_j x_{i} + b_j }{||w||} y^{i} )$$
到样本最近距离取最大，这就是 SVM 的任务。我们现在用数学公式表达了 SVM 的任务，就是找到距离分割平面点的间隔最大值。

那么我们现在将问题扩展到 n 维特征空间，$W^TX + b $ 这里 W 和 X 是 n 维，
$$d = \frac{|W^T + b|}{||w||}$$ 

$$ y(x) = w^T \Phi(x) + b $$
这里解释一下$\Phi$ 主要是样本的特征一种映射，将样本$x^i$ 的 $ (x_1,x_2,x_3) $ 三维特征映射到下面多维特征向量
$$ \Rightarrow (1,x_1,x_2,x_3,x_1^2,x_2^2,x_3^2,x_1x_2,x_2x_1,x_3x_1,x_3x_2)$$
将原始特征通过$\Phi$就变成了更多的特征，对数据的特征进行可能特征映射。做一阶$\Phi$ 就是特征向量本身。
求解分割平面问题其实就是凸二次规划问题

### 推导目标函数
$$ y(x) = w^T \Phi(x) + b$$
其中y(x) 表示第i样本估计值，我们知道位于分割面法线方法为y(x)为正那么其真实值为1 ,f(x) 和 y 相乘为正，反之亦然。
$$ \begin{cases}
    f(x_i) > 0 \Leftrightarrow y_i = 1 \\
    f(x_i) < 0 \Leftrightarrow y_i = -1 
\end{cases} \Leftrightarrow y_i f(x_i) > 0$$
下面是点到直线距离公式
$$ \frac{y_i \cdot f(x_i)}{||w||} = \frac{y_i \cdot (w^T \cdot \Phi(x_i) + b)}{||w||}$$

下面公式 arg 表示对于$y_i \cdot (w^T \cdot \Phi(x_i) + b)$ 这样距离对所有点求最近在求最远。
$$ arg \max_{w,b} \{  \min_i \frac{1}{||w||}[y_i \cdot (w^T \cdot \Phi(x_i) + b)]  \}$$
$$ arg \max_{w,b} \{ \frac{1}{||w||} \min_i [y_i \cdot (w^T \cdot \Phi(x_i) + b)]  \}$$
我们需要求 w 和 b 来满足上面目标函数，怎么优化是比较麻烦，
- $w = (w_1,w_2, \dots w_k)$
- $||w|| = (w_1,w_2, \dots w_k)(w_1,w_2, \dots w_k)^T$
#### 化简目标函数
其实我们这些支持向量点到分割平面一定是一个参数假设是 C 那么他们距离表示是将这些支持向量点带入上面点到直线方程，因为可以对直线做线性变换除以一个常量 C 
$$\frac{y_i  (w^T  \Phi(x_i) + b)}{||w||} = C$$
$$\frac{y_i  (w^T  \Phi(x_i) + b)}{C||w||} = 1$$
w 向量乘上常数 C 后方程是没有变化的。等比例缩放 w 总是可以办到。就可以将距离直线距离取 1。
总可以通过等比缩放w方法，使得函数值满足 $|y \ge 1|$
$$ y_i  (W^T  \Phi(x_i) + b ) \ge 1$$
如果满足上面条件那么$ y_i  (W^T  \Phi(x_i) + b ) $最小值就是 1 将 1 带入上面方程
$$ \arg \max_{w,b} \frac{1}{||w||} $$
$$ s.t. y_i  (W^T  \Phi(x_i) + b ) \ge 1 , i = 1,2, \cdots , n $$

$$ \max_{w,b} \frac{1}{||w||} $$
$$ s.t. y_i  (W^T \Phi(x_i) + b ) \ge 1 , i = 1,2, \cdots , n $$

$ \max_{w,b} \frac{1}{||w||} $ 就等价于 $ \min_{w,b} ||w||^2 $

$$ \Rightarrow \min_{w,b} \frac{1}{2}||w||^2 $$
$$ s.t. y_i  (W^T  \Phi(x_i) + b ) \ge 1 , i = 1,2, \cdots , n $$


#### 
目标函数是有约束，有 N 样本个数，那么我们有 N 个约束，这就是我们变化为，在约束条件下求最小，拉格朗日乘子法
$$ \min f(x) = $$ 是在约束条件下，我们可以将所有约束条件转化小于 0  如下，如果不是小于 0 就在不等式两边取负号如果没有等于 0 我们就追加等式条件。那么我们的约束条件就转化为有 n 小于 0 的和 m 个等于 0 的约束条件。
$$ \begin{cases}
    f_1(x) \le 0 \\
    f_2(x) \le 0 \\
    \cdots \\
    f_n(x) \le 0 
\end{cases}$$
$$ \begin{cases}
    h_1(x) = 0 \\
    h_2(x) = 0 \\
    \cdots \\
    h_m(x) = 0 
\end{cases}$$
所有约束条件，

$$ \mu_1 \ge 0 \mu_2 \ge 0 \cdots \mu_n \ge 0 $$
$$ \lambda_1 = 0,\lambda_2 = 0 \dots \lambda_m = 0 $$
通过在原始函数f(x) 添加乘子$\sum_{i=1}^n \mu_i f_i(x)$和$\sum_{j=1}^m \lambda_j h_j(x)$我们就得到了拉格朗日函数，
$$ G(x,\vec{\mu},\vec{\lambda}) = f(x) + \sum_{i=1}^n \mu_i f_i(x) + \sum_{j=1}^m \lambda_j h_j(x) $$

![屏幕快照 2019-11-20 上午5.19.18.png](https://upload-images.jianshu.io/upload_images/8207483-0c096452bd22b7e4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可以将方程进行变化理解有关$\mu_i$ 的线性方程，将$\mu_i$移动方程一边其他项看做方程的常数项就可以得到线性方程，因为$\mu$ 是一元线性的，假设上图L1,L2,L3 分别都是 $\mu$ 的线性方程。然后将min(l1,L2,L2) 取最小值点拦截后就得到一个凸函数(浅红线表示)。凹函数存在局部最大值也就是全局最大值。max min(l1,l2,...ln) 
$$ G(x,\vec{\mu},\vec{\lambda}) = f(x) + \sum_{i=1}^n \mu_i f_i(x) + \sum_{j=1}^m \lambda_j h_j(x) $$
$$s.t. f_i(x) \le 0 h_j(x) = 0$$
我们考虑一下要是对其$ max_{\mu,\lambda,\mu_i \ge 0} G(x,\vec{\mu},\vec{\lambda})$
因为$\mu_i \ge 0 $ 乘以 $f_i(x) \le 0 $ 一定是一个负数，因为$h_j(x) = 0$,那么我们就得到$ \max_{\mu,\lambda,\mu_i \ge 0} G(x,\vec{\mu},\vec{\lambda}) = f(x)$
那么我么原始问题求$\min f(x)$ 就变成了
$$ \min_x (\max_{\mu,\lambda,\mu_i \ge 0} G(x,\vec{\mu},\vec{\lambda})) $$

$$ \max_{\mu,\lambda,\mu_i \ge 0} (\min_x G(x,\vec{\mu},\vec{\lambda})) $$
#### 支持向量点
在这里我们尝试用一种思维来解释什么是支持向量点，（下图）不难看出穿过支持向量点超平面是平行于分割平面的。根据线性变化我们可以进行一些假设。假设分割平面的 w 就是由这些距离分割平面的点的乘上系数而得，因为 x 维度和 w 维度是一样，可以想象
$$\vec{w} = \sum_{i=1}^k \alpha x^{support vector}$$
![支持向量点](https://upload-images.jianshu.io/upload_images/8207483-82559b74d67dc3ac.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们将支持向量点的间距离称为间隔，在间隔区域没有任何数据点，可以将其想象一个安全地带，我们尽量找到让间隔最大（安全地带）的分割平面。

- 向量 $\vec{w} = [w_1,ws]$ 是分割超平面的法线方向
- $wx + b = 0$ 和 $wx' + b = 0 \Rightarrow w(x' - x ) = 0$ 正交
- $

$$c(x,y,f(x)) = \begin{cases}
    0, & y*f(x) \ge 1 \\
    1-y*f(x) & 
\end{cases}$$
目标函数
$$ min_w \lambda ||w||^2 + \sum_{i=1}^n (1 - y_i(x_i,w))_{+} $$

As you can see our objective of a SVM consists of two terms,The first term is a regularizer, the heart of SVM the second term the loss.The regularizer balances between margin maximization and loss, we want to find the decision surface that is maximally for away from any data points

$$ \frac{\partial }{\partial W_k} \lambda ||w||^2 = 2 \lambda W_k$$
$$ \frac{\partial}{\partial W_k} (1 - y_i(x_i,w))_{+} = \begin{cases}
    0 \\
    -y_i x_{ik} 
\end{cases} $$

$$ w = w + \eta(y_i x_i - 2 \lambda w) $$

我们假设一个方程为 $f(x) = \vec{w}^T  \vec{x} + b $ 为了方便以后我们简化写完$f(x) = wx + b$ 不少书籍上也是这么写的。

### 线性 SVM()
就是找到一条分开两类事物的一个超平面，如果在有一个特征值就是一条分割线，而在两个特征值情况下就是一个分割平面，在 N 个特征中就是分割的超平面。如图这里有

, 一个普通的SVM就是一条直线罢了，用来完美划分linearly separable的两类。但这又不是一条普通的直线，这是无数条可以分类的直线当中最完美的，因为它恰好在两个类的中间，距离两个类的点都一样远。而所谓的Support vector就是这些离分界线最近的『点』。如果去掉这些点，直线多半是要改变位置的。可以说是这些vectors（主，点点）support（谓，定义）了machine分类器）...



## 前言
当下机器学习比较重要 3 中算法，个人都目前为止认为比较重要机器学习算法分别是，深度学习、SVM 和决策树。在深度学习出现之前，是 SVM 的时代 SVM 占据了机器学习算法优势地位整整 15 年。
在今天在深度神经网打压下，SVM 还是有自己一席之地。

### 概念
- 线性可分支持向量机
- 线性支持向量机
- 非线性支持向量机
    - 核函数

![svm_01.png](https://upload-images.jianshu.io/upload_images/8207483-e812b58deb8c85b0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- $ \vec{w} \cdot \vec{b} - b = 0 $ 就是对应于**决策边界**
-  支持向量 $ d_1, d_2, d_3 $ 也就是支持向量机点到决策边界距离要小于其他样本点到决策边界的距离。
#### 点到直线距离
$$ Ax + By + C = 0$$
$$ f(x_0,y_0) = \frac{|Ax_0 + By_0 + C|}{\sqrt{A^2 + B^2}} $$
$$ \frac{A}{\sqrt{A^2 + B^2}} x_0 + \frac{B}{\sqrt{A^2 + B^2}} y_0 + \frac{C}{\sqrt{A^2 + B^2}}$$

### 目标和原理



### 计算过程
#### 分割平面，
$$ f(x) = w \cdot x + b $$
w 是样本线性组合而成
$$ \vec{w} = \sum_{i=1}^N \alpha_i x^{(i)} y^{(i)} $$
#### 点到直线距离
$$ Ax + By + C = 0$$
$$ f(x_0,y_0) = \frac{|Ax_0 + By_0 + C|}{\sqrt{A^2 + B^2}} $$
### 软间隔
- 对线性不可分的数据给出()
### 核函数

### SMO

### 用数学语言描述 SVM
$$ X = {(x_1,y_1),(x_2,y_2) \dots (x_i,y_i)} \mathbb{R}$$
$$ dist_i = y_i (\frac{w}{||w|| \dot x_i + \frac{b}{||w||}})$$
样本点到超平面的几何间隔的最小值为
$$ dist_{total} = \min_{i=1,2, \dots, N} dist_i$$

### 最小和最大
- 最小
- 最大

## SVM VS 深度学习
深度学习是对原始特征重新表示，也就是原始特征虽然学的很懒，通过。就拿深度学习擅长的图像识别，是因为我们人类使用 pixel 来表示图片，用 pixel 来表示图片是一种比较烂表示方式，这是通过深度学习能够帮助我提取一些新表示特片特征来对图像进行表示。
而且深度学习有时候即使对于设计者是个黑核模式，也就是我们不知道机器如何学习来给出其预测或推荐。有时候我们是需要知道机器是如何学习，学习到了什么，以及如何做出决定和推荐。

而 SVM 好处是通过数学推导出来的模型。个人尝试了解一下感觉还是比较难。很多机器学习框架都提供了 SVM 的支持，我们只需要调用方法就可以完成 SVM 建模型过程。但是如何设置合理参数来获得较高精度就需要我们对参数用途比较了解。
## SVM VS 决策树
SVM 和决策树都是找决策边界

## SVM(Support Vector Machine)

SVM = Hinge Loss + (核方法)Kernel Method

### 概念 SVM (Support Vector Machine)

SVM 是我们必须掌握统计算法，而且 SVM 推导过程要比其应用更加重要。使用 SVM 前提是我们分类问题是线性可分的。SVM 既可以解决分类问题也可以解决回归问题。

将 SVM(支持向量机)思想转换为最优化的问题。

SVM 用于解决分问题的分类器。介绍且背后算法，了解算法后便于我们更好调整参数。在深度学习出现之前被大家广泛应用用于分类算法。
![svm](https://upload-images.jianshu.io/upload_images/8207483-4d939008a9e07a26.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
图中有我们用颜色来表示不同样本，所谓分类问题就是找到一条线，让线两边分别是不同的类别。$L_1$ 

我么不仅需要关注训练误差，我们更要关系期望损失
### SVM 关键术语
- 间隔: 就是离
- 对偶
- 核技巧:在 SVM 出现之前就有核技巧，通过核技巧让 SVM 从欧式空间扩展到多维空间
### SVM 分类
- 硬间隔 SVM
- 软间隔 SVM
- 核 SVM

### SVM 推导过程
- 监督学习
我们对比深度学习 SVM，深度学习通过梯度下降不断优化边界，然后最后找到最优解而 SVM 无需梯度下降就可以
- 几何含义
就是我们要找的决策分界线到离其最近点的边距最大。这就是 SVM 的几何含义。那么有了这个想法我们就需要用数学来描述我们的想法，这也就是找模型的过程。
- 决策
![support-vector-machine-and-implementation-using-weka-19-638.jpg](https://upload-images.jianshu.io/upload_images/8207483-98fbf8be1a4a64e2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我们绘制 $\vec{w}$ 垂直于我们决策边界的向量，然后计算一点 $\vec{u}$ 向量到  $\vec{w}$ 上的投影，如果投影值大于一个常数值就说明是正样本否则就是负样本。
$$ \vec{w} \cdot \vec{u} \ge C then u \in + $$
$$ \vec{w} \cdot \vec{u} - b \ge 0 then u \in + $$
### 训练数据
我们在构建训练数据集时候，需要将所有正样本设计为计算后大于 1
$$ \vec{W} \cdot \vec{X_+} + b \ge 1$$
$$ \vec{W} \cdot \vec{X_-} + b \le -1$$
这就是 SVM 的最大间隔假设。
### 小技巧
我们通过一些小技巧将上面两个公式合并为一个公式
$$ y_i(\vec{w}x_+ \cdot \vec{u} + b) \ge 1$$
$$ y_i(\vec{w}x_- \cdot \vec{u} + b) \ge 1$$
$$ y_i(\vec{w}x \cdot \vec{u} + b) \ge 1$$
约束条件在训练集中所有样本都需要满足这个公式
对于支持向量的点在上面公式不等号变为等号
$$ y_i(\vec{w}x \cdot \vec{u} + b) = 1$$
### 求取宽度
$$ width = (\vec{x_+} -\vec{x_-}) \cdot \frac{\vec{W}}{||w||} $$

$$ width = (\vec{x_+} -\vec{x_-}) \cdot \frac{\vec{W}}{||w||} = \frac{\vec{x_+} \cdot \vec{w}}{||w||} - \frac{\vec{x_-} \cdot \vec{w}}{||w||}$$
$$ width = \frac{1-b}{||w||} + \frac{1+b}{||w||} = \frac{2}{||w||}$$
$$ y_i(\vec{w}x \cdot \vec{u} + b) = 1$$
那么我们让街宽最大就是让 $\frac{2}{||w||}$ 大，那么也就是问题变为了$\max \frac{2}{||w||}$
我们将问题进行一系列转换那么求 $\max \frac{2}{||w||}$ 转换为了 $ \min \frac{1}{2} ||w||^2$

$$ L = \frac{1}{2} ||\vec{w}||^2 - \sum \alpha_i (y_i(\vec{w} \cdot \vec{x} + b ) - 1)  $$

$$ \frac{\partial L}{\partial \vec{w}} = \vec{w} - \sum \alpha_i \cdot y_i \cdot x_i = 0 $$
$$  \vec{w} = \sum \alpha_i \cdot y_i \cdot x_i  $$

$$ \frac{\partial L}{\partial b} = \sum \alpha_i y_i  = 0 $$
然后将 $  \vec{w} = \sum \alpha_i \cdot y_i \cdot x_i  $ 公式带入下面公式中进行推导
$$ L = \frac{1}{2} ||\vec{w}||^2 - \sum \alpha_i (y_i(\vec{w} \cdot \vec{x} + b ) - 1)  $$

$$ L = \frac{1}{2}  \sum \alpha_i \cdot y_i \cdot x_i \vec{w}  \sum \alpha_i \cdot y_i \cdot x_i - \sum \alpha_i y_i x_i \cdot (\sum \alpha_i y_i x_i )- \sum \alpha_i y_i b + \sum \alpha_i $$
$$ = \sum\alpha_i - \frac{1}{2} \sum \sum \alpha_i y_i x_i \alpha_j y_j x_j$$

### Hinge Loss
1. 找到函数模型集合
这里我们依旧是进行二分类问题，我们在训练集样本 $\{ x^1 x^2 \cdots x^i \}$ 对于标签为 $\{ \hat{y}^1 \hat{y}^2 \cdots \hat{y}^n \}$ 这里用-1 和 1 表示结果。
$$ g(x)=\begin{cases}
f(x) > 0,Output = 1 \\ 
f(x) < 0,Output = -1
\end{cases} $$
定义 g(x) 函数，其中包含一个函数 f(x) 当 f(x) > 0 g(x) 为 1 反之为 -1。
2. **损失函数**
$$ L(f) = \sum_n \delta(g(x^n) \neq \hat{y}^n ) $$
对于所有数据 $\delta$ 表示内部表达式为真则为 1 否则为 0。那么当 g(x) 不等于 $\hat{y}^n$ 时候输出为 1 ，损失函数计算函数就记录一次，求和后会记录犯了几次错误。但是很明显这个函数无没有办法微分的。
$$ L(f) = \sum_n \hat{y}^n f(x)$$
![svm_1](https://upload-images.jianshu.io/upload_images/8207483-010bfd7f38821f2f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- Square Loss
$$ if  \hat{y}^n = 1 , f(x) close to 1 $$
$$ if  \hat{y}^n = -1 , f(x) close to -1 $$
$$ l(f(x^n),\hat{y}^n) = (\hat{y}^nf(x^n) - 1)^2$$
然后我们把 $\hat{y} = 1 $ 带入方程得到 $(f(x^n) - 1)^2$，也就是说明当  $\hat{y} = 1 $  时候 $f(x^n)$ 越接近 1 也好。
然后我们把 $\hat{y} = -1 $ 带入方程得到 $(f(x^n) + 1)^2$，$\hat{y} = -1 $  时候 $f(x^n)$ 越接近 -1 也好。
- Sigmoid + Square Loss
$$ if  \hat{y}^n = 1 , f(x) close to 1 $$
$$ if  \hat{y}^n = -1 , f(x) close to -1 $$
$$ l(f(x^n),\hat{y}^n) = (\hat{y}^nf(x^n) - 1)^2$$


### SVM 推导
$$ f(x) = W^Tx + b $$
根概率没有关系，
$$ f(w) = sign(w^Tx + b) $$
- 几何含义
就是我们要找的决策分界线到离其最近点的边距最大。这就是 SVM 的几何含义。那么有了这个想法我们就需要用数学来描述我们的想法，这也就是找模型的过程。我们不仅关注训练误差更关注于期望误差。例如这个线对噪声非常敏感，泛化能力不高。
### SVM 分类
- 硬间隔 SVM: 最大就是间隔期(max margin W b)
- 软间隔 SVM
- 核 SVM

#### 硬间隔 SVM
接下来我们做的工作就是用形式语言(数学语言)来翻译这句话**最大化间距**
- 假设 $\{ (x_i,y_i)\}^N_{i=1} $ 其中 $ x_i \in \mathbb{R} ,y \in \{-1,1\}$

$$ st. \begin{cases}
    w^Tx_i + b > 0 & y_i = 1 \\
    w^Tx_i + b < 0 & y_i = -1 
\end{cases} $$
等价于下面等式，因为$y_i$ 和 
$$ y_i(w^Tx_i + b) > 0$$
现在我们就用数学语言将最大分类器进行说明。
所谓 margin 就是距离分割平面最小距离的点为 margin，最小 min 距离
$$ margin(w,b) = \min_{w,b,x_i} distance(w,b,x_i)$$

$(x_i,y_i)$点到直线 $w^Tx + b$ 的距离公式如下 

$$ distance = \frac{1}{||w||} |w^Tx_i + b| $$
$$ margin(w,b) = \min_{w,b,x_i,i \in (1 \rightarrow N)} \frac{1}{||w||} |w^Tx_i + b|$$

然后我们将 distance 带入到公式，还需要满足 $y_i(w^Tx_i + b) > 0$ 条件
$$ \begin{cases}
    \max_{w,b} \min_{x_i} \frac{1}{||w||} |w^Tx_i + b| \\
    s.t. y_i(w^Tx_i + b) > 0
\end{cases} $$

因为$y_i(w^Tx_i + b)$ 是大于 0 而且 $y_i$ 取值为 1 ，所以可以将其带入上面公式来脱掉绝对值。
$$ \begin{cases}
    \max_{w,b} \min_{x_i} \frac{1}{||w||} y_i(w^Tx_i + b) \\
    s.t. y_i(w^Tx_i + b) > 0
\end{cases} $$

$$ max_{w,b} \frac{1}{||w||} min_{x_i} y_i(w^Tx_i + b) $$
我们假设有一个大于 0 的 $\gamma$ 等于该方程
$$ s.t. y_i(w^Tx_i + b) >0 \Rightarrow \exists \gamma>0, min_{x_i} y_i(w^Tx_i + b) = \gamma $$

$$ \Rightarrow \begin{cases} max_{w,b} \frac{1}{||w||} \\ min y_i(w^Tx_i + b) =1 \end{cases} \Rightarrow \begin{cases}
    min \frac{1}{2} w^Tw \\
    s.t. y_i(w^Tx_i + b) \ge 1
\end{cases} $$
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

- 点 $ (x,y)$ 到 $Ax + By + C = 0$ 的距离 
$$\frac{|Ax + By + C|}{\sqrt{A^2 + B^2}} $$

- 扩展到 n 维空间 $\theta^Tx_b = 0 \Rightarrow w^T + b = 0$

$$ \frac{|w^T + b|}{||w||} ||w|| = \sqrt{w_1^2 + w_2^2 \cdots w_i^2}$$
我们了解到了如何在 n 维空间进行求解点到线或平面的距离后，我么知道所有点到平面的距离都应该大于支持向量到平面距离，然后接下来我们再尝试用数学方式把这些思想表达出来。 

这里对于分类问题使用 1 和 -1 表示两类事物，而非 0 和 1。
$$ \begin{cases}
    \frac{w^Tx^{(i)} + b}{||w||} \ge d & \forall y^{(i)} = 1 \\
    \frac{w^Tx^{(i)} + b}{||w||} \le -d & \forall y^{(i)} = -1
\end{cases}$$
通过公式不难看出对于任意样本点 $y^i = 1$ 都满足 $\frac{w^Tx^{(i)} + b}{||w||} \ge d$
对等式两边分别除以 d 就得到下面不等式
$$ \begin{cases}
    \frac{w^Tx^{(i)} + b}{||w||d} \ge  & \forall y^{(i)} = 1 \\
    \frac{w^Tx^{(i)} + b}{||w||d} \le -1 & \forall y^{(i)} = -1
\end{cases}$$

这里 $||w||$ 是 n 维的向量的模是一个数字，d 也是数，我们可以对 w 和截距 b 同时除以一个数。转换成下面方程
$$ \begin{cases}
    w^T_dx^{(i)} + b_d \ge 1 & \forall y^{(i)} = 1 \\
    w^T_dx^{(i)} + b_d \le -1 & \forall y^{(i)} = -1
\end{cases}$$

那么我们这里方程中有两个未知数 $W_d$ 和 $b_d$ 需要我们求解，这样我们就可以使用 w 和 d 直接进行替换。但是现在使用 w 和 d 和之前 w 和 d 差一个系数关系。

我们在进一步进行推导出，这样我们将两个不等式合并表示为一个不等式。也就是说明我们所有点都要满足这个不等式关系。
$$ y^{(i)}(w^Tx^{(i)} + b) \ge 1$$


推导到现在我们发现决策边界线可以表达为 $$ W_d^T + b = 0 $$
而其上下两侧的支持向量的直线可以用 $ W_d^T + b = 1 $ 和 $ W_d^T + b = -1 $
对于任意支撑支持向量
$$ \max \frac{|w^Tx+b|}{||w||} \Rightarrow \max \frac{1}{||w||} \Rightarrow \min ||w|| \Rightarrow \min \frac{1}{2} ||w|| ^2$$
经过一些列推导我们得到最小值，求取最小值也就是我们问题变为可以优化的问题。不过这一切是建立在满足以下不等式基础上
$$s.t.  y^{(i)}(w^Tx_i + b) \ge 1$$
之前我们讨论最优化条件都是，


## 软间隔(soft margin)
我们在之前推导硬间隔(hard margin)
$$ \min \frac{1}{2} ||w||$$
$$ s.t. y^{(i)}(W^Tx^{(i)} + b) \ge 1$$

由于一些误差点或者是线性不可分情况,这时候我们需要设计一个具有容错能力 SVM 就可以解决这个问题。

$$ \min \frac{1}{2} ||w|| $$
$$ s.t. y^{(i)}(W^Tx^{(i)} + b) \ge 1$$
我们知道下面约束就会保证任一点都在直线大于 1 或 -1 以外的空间。
$$ s.t. y^{(i)}(W^Tx^{(i)} + b) \ge 1 - \zeta_i$$
![soft-margin](https://upload-images.jianshu.io/upload_images/8207483-a0c4f53c23d7c137.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
$$ s.t. y^{(i)}(W^Tx^{(i)} + b) \ge 1 - \zeta_i$$
$$ \zeta \ge 0 $$
$$  f(x) = W^Tx + b $$
$$ V^i = \frac{W^T}{||W||} x^i + \frac{b}{||W||}$$
$$ V^i = \frac{W^T}{||W||} x^i + \frac{b}{||W||}$$

### 线性 SVM
#### 找出方程组
$$ f(x) = \sum_i w_ix_i + b = \left[ \begin{matrix}
    w \\ 
    b
\end{matrix} \right] \cdot \left[ \begin{matrix}
    x \\ 
    1
\end{matrix} \right] = W^T x$$

#### 损失函数

#### 梯度下降

### 一般步骤
- 数据准备
- 升维度找到支持向量分类器
- 

