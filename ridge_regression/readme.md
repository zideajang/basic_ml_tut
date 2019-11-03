## 岭回归(Ridge Regression)

> 图片和内容引用 statQuest with Josh Starmer 但是内容是在自己理解基础上分享所有特此声明


需要看一下标准方程方法
$$ w = (X^TX)^{-1}X^Ty $$

- 通过一个简单例子让大家了解岭回归背后原理
- 深入了解岭回归的工作机制
- 岭回归在变化条件下是如何工作
- 岭回归相对其他优化算法其自身优点所在

我们从简单线性回归入手，从图(图1-1)上可以明显地查看 size 和 weight 的线性关系
![图1-1](https://upload-images.jianshu.io/upload_images/8207483-c434b5c72afea547.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可以使用**最小二乘法**来找到拟合这些训练集点的直线
![图1-2](https://upload-images.jianshu.io/upload_images/8207483-eafb2f6c1461c6a6.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

计算得出 $y=0.9 + 0.75x$ 其中 0.9 为j截距(偏置) 0.75 为权重

不过如果我们训练集数据量过小，只有两个点，这时候我们知道两点确定一条直线，所以通过计算损失函数我们能够找到一条直线经过这两个点如图(图1-3)
![图1-3](https://upload-images.jianshu.io/upload_images/8207483-7eedb7b1c97f18cc.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果我们将测试集数据(用绿色表示)，引入到图中就会发现虽然我们模型很好拟合训练集数据（红色）他却无法拟合到测试集中的数，引入岭回归就是为了解决这个过拟合问题。
![图1-5](https://upload-images.jianshu.io/upload_images/8207483-0337c0ac3d2db06e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![图1-6](https://upload-images.jianshu.io/upload_images/8207483-2c220a5e41e17543.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如图我们通过岭回归是找到一条直线，这条直线虽然没有在训练集上有很好表现，但是可以更好表现在测试集上，我们通过添加一些干预（或者叫噪音）来放置过拟合现象发生。
![图1-7](https://upload-images.jianshu.io/upload_images/8207483-a1b59f8acc47e3d6.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 
![图2-1](https://upload-images.jianshu.io/upload_images/8207483-1cbc520f6d4a8640.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
****
假设通过最小二乘法进行损失函数值为 0 时的模型为 $y = 0.4 + 1.3x $ 
引入 $ \lambda w^2$ 这个惩罚值后 $y = 0.5 + 1.1x + \lambda slope^2$ 
假设 $\lambda = 1$ 这样原有损失函数值变为 
$$ = 0 + 1 * 1.3^2 = 1.69$$
让现在损失函数值变为 1.69 ,也就是拟合较好直线通过
调整参数根据后求的一个新的解模型变为
$$ y = 0.9 + 0.8 x$$ 
这个模型虽然没有完全拟合训练集上的点，岭回归损失函数值变为 $0.3^2 + 0.1^2 + 1 * 0.8^2= 0.74 $
![图2-2](https://upload-images.jianshu.io/upload_images/8207483-654e71c1c25c17f2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![图2-5](https://upload-images.jianshu.io/upload_images/8207483-1010b849684562df.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![图2-3](https://upload-images.jianshu.io/upload_images/8207483-0b28525187156bbb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
通过对比我们不难发现引入了岭回归的损失函数在测试集上有更好的表现。

![图2-7](https://upload-images.jianshu.io/upload_images/8207483-095f5bfea01339ba.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![图2-6](https://upload-images.jianshu.io/upload_images/8207483-32e5969ac3c985ff.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从而不难看出当直线斜率较小时候，我们 y 估计值对 x 敏感程度较低。我们岭回归正式利用这点来减缓 x 对为 y 影响程度。

可以尝试了一组 lambda 的值，并使用交叉验证来确定哪个 lambda 值效果最好



通过对比明显感觉通过岭回归的得到模型要好于之前简单模型。

如果训练样本 m 数量少，甚至少于样本特征点 n，这样将导致数据矩阵$(X^TX)^-1$ 不是满秩矩阵，而无法求逆。

为了解决这个问题，统计学家引入了**岭回归**的概念

$$ w = (X^TX + \lambda I)^{-1}X^Ty $$

$\lambda$为领系数，I为单位矩阵(对角线上全为 1 其他元素为0)

$$ J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_{\theta}(x_i) - y_i)^2 + \lambda \sum_i^n \theta_i^2 $$

$$ J(\theta) $$