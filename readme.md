## 机器学习基础篇

### 使用库
- sklearn
### 监督学习
### 非监督学习

## 特征缩放和交叉验证法
$$x_1 = 1000000 cm^2 - 2000000 cm^2 \tag{area}$$
$$x_2 = 1 - 5 \tag{room count}$$
因为特征间数据取值范围相差过大，就会造成梯度下降会走的很远。这样优化时间比较长而且可能造成错误路径。
### 数据归一化
- 就是把数据的取值范围处理为 0 - 1 或者 -1 1 之间
    - 任意数据转化为 0 - 1 之间 ( $ newValue = \frac{oldValue - min}{max - min}) $
    - 任意数据转化为 -1 - 1 之间  ( $ newValue = \frac{oldValue - min}{((max - min) - 0.5) * 2} )$
### 均值标准化
- x 为特征数据，u 为数据的平均值，s 为数据的方差
$$ newValue = \frac{oldValue - u}{s}$$
- 取值范围从 -0.5 - 0.5

### 交叉验证法
- 通常我们会将数据集按一定比例进行切分为训练数据集和测试数据集
- 对于较小数据集时候我们就会用到交叉验证法
#### 交叉验证法做法
- 把所有的数据切分为 10 份，如果有 100 样本切分每个数据集有 10 个样本

$$ E = \frac{1}{10} \sum_{i=1}^{10} E_i $$

## 拟合
![拟合](https://upload-images.jianshu.io/upload_images/8207483-75dea78d54022e22.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 欠拟合(Underfitting)
- 正确拟合(good fitting)
- 过拟合(overfitting)

![分类问题拟合](https://upload-images.jianshu.io/upload_images/8207483-3fe71dd2041b94e8.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这两个点可能是噪声数据，分类边界比较复杂，依旧在训练集表现优秀而在测试集就会发现错误率就高于训练集的错误率。

### 防止过拟合
本质原因就是选择了复杂模型来处理简单问题
- 较少样本的特征
- 增加数据量，一般情况增加数据量会提升训练的效果
- 正则化(Regularized)
![optimal_capacity.png](https://upload-images.jianshu.io/upload_images/8207483-ace52e8f287b78df.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 模型容量
我们模型容量过小
- 奥卡姆剃刀定律
### 正则化方法
- 数据，提高数据质量和数量，人工数据合成，在训练数据加入造成。
- 模型
- 优化，设计抵抗过拟合优化方法

### 正则化损失函数
正则化用于防止过拟合，就是直接修改成本函数
也就是我们不仅关注局部而且还要关注全局。
就是我们损失函数添加一个，通过$\lambda$来调节正则化的重要性。就是把所有参数$\theta$ 进行累加。
- L2 正则化
限制参数范围（参数接近零）
$$ J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2 \right]$$
- L1 正则化
稀疏参数空间(少量非零参数)
$$ J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n |\theta_j| \right]$$

### 将模型容量
- 参数共享，例如卷积神经网络，其实背后是有生物学的依据
- 系综方法，例如 dropout 可以达到降低模型容量
- 多任务学习
- Batch Normalization
- 非监督/半监督学习
- 生成式对抗网络 


## 聚类算法
### 聚类算法 K-Means 算法



## 深度神经网咯
### 循环神经网
#### 什么是神经网
#### 当下流行神经
#### 什么是循环神经网
在使用 google 进行搜索时候，当我们输入要搜索内容，google 会根据我们喜欢提出自动补全的文字，帮助我们快速找到要查询的内容。
#### RNN 内部机制
#### 梯度消失和梯度爆炸
#### 长短期循环神经网
#### 实例