![线性回归数据](https://upload-images.jianshu.io/upload_images/8207483-f3b9a011628467ac.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
准备数据
```python
#coding=utf-8
import numpy as np
def run():
    # 第一步 准备数据
    points = np.genfromtxt('data/data.csv',delimiter='.')
if __name__ == '__main__':
    run()
```
定义模型
```python
#coding=utf-8
import numpy as np
def run():
    # 第一步 准备数据
    points = np.genfromtxt('data/data.csv',delimiter='.')

    # 第二步 定义超参数
    learning_rate = 0.0001 #更新模型的频率
    init_w = 0 #定义权重(一元线性方程斜率)
    init_b = 0 #定义偏置值(一元线性方程的截距)
    mun_iterations = 1000
    
if __name__ == '__main__':
    run()
```
训练模型
```python
def run():
    # 第一步 准备数据
    points = np.genfromtxt('data/data.csv',delimiter=',')

    # 第二步 定义超参数
    learning_rate = 0.0001 #更新模型的频率
    w = 0 #定义权重(一元线性方程斜率)
    b = 0 #定义偏置值(一元线性方程的截距)
    mun_iterations = 1000

    #训练模型
    print 'starting gradient descent at b = {0}, w ={1}, error ={2}'.format(init_b,init_w,cal_error(init_b,init_w,points))
    [b,w] = gradient_descent_runner(points,init_b,init_w,learning_rate, mun_iterations)
    print 'ending gradient descent at b = {0}, w ={1}, error ={2}'.format(b,w,cal_error(b,w,points))



if __name__ == '__main__':
    run()
```
定义计算误差函数
$$ Error_{(w,b)} = \frac{1}{N} \sum_{i=1}^N (y_i - (mx_i + b))^2 $$

```python
def cal_error(b,w,pts):
    # 初始化误差
    totalError = 0
    for i in range(0, len(pts)):
        x = pts[i,0] #获取 x
        y = pts[i,1] #获取 y
        # 计算所有样本点估计值到期望值间距离的平方差和
        totalError += (y - (w * x + b)) ** 2
    # 取平均值
    return totalError / float(len(pts))
```

```python
def gradient_descent_runner(pts,starting_b,starting_w,learning_rate,mun_iterations):
    # 接受参数
    b = starting_b
    w = starting_w

    for i in range(mun_iterations):
        # 每一次迭代都会更新 b 和 w 
        b, w = step_gradient(b,w,np.array(pts),learning_rate)
    return [b,w]
```

$$ \frac{\partial L(w,b)}{\partial w} = \frac{2}{N} \sum_{i=1}^N -x_i(y_i - (wx_i + b)) $$
$$ \frac{\partial L(w,b)}{\partial b} = \frac{2}{N} \sum_{i=1}^N -(y_i - (wx_i + b)) $$

```python
# 定义每一次迭代
def step_gradient(b_current, w_current, pts, learningRate):
    b_gradient = 0 # b 变化速度
    w_gradient = 0 # w 变化速度
    # 获取样本数
    N = float(len(pts))
    for i in range(0, len(pts)):
        x = pts[i, 0]
        y = pts[i, 1]
        # 分别计算损失函数对于 b 和 w 的偏导数来作为 b 和 w 变化速度
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    # 根据导数(导数也就损失函数下降率)和学习率来决定调整参数幅度(步长)
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]
```

```
ending gradient descent at b = 0.0889365199374, w =1.47774408519, error =112.614810116
(base) jangwoodeMacBook-Air:basic_regression jangwoo$ python simple_linear_regression.py
starting gradient descent at b = 0, w =0, error =5565.10783448
ending gradient descent at b = 0.0889365199374, w =1.47774408519, error =112.614810116
```

### 多元线性回归
$$ price = w_1x_1 + w_2x_2 + w_3x_3$$
$x_1$ 表示房子的面积
$x_2$ 表示房子的卧室数量
$x_3$ 表示房子的芳龄

准备数据集

```python
#coding=utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('./data/homeprices.csv')
print df
```

```
   area  bedrooms  age   price
0  2600       3.0   20  550000
1  3000       4.0   15  565000
2  3200       NaN   18  610000
3  3600       3.0   30  595000
4  4000       5.0    8  760000
5  4100       6.0    8  810000
```

**回归**是由英国著名生物学家兼统计学家高尔顿(Francis Galton,1822～1911.生物学家达尔文的表弟)在研究人类遗传问题时提出来的。为了研究父代与子代身高的关系，高尔顿搜集了1078对父亲及其儿子的身高数据。他发现这些数据的散点图大致呈直线状态，也就是说，总的趋势是父亲的身高增加时，儿子的身高也倾向于增加。但是，高尔顿对试验数据进行了深入的分析，发现了一个很有趣的现象—回归效应。因为当父亲高于平均身高时，他们的儿子身高比他更高的概率要小于比他更矮的概率；父亲矮于平均身高时，他们的儿子身高比他更矮的概率要小于比他更高的概率。它反映了一个规律，即这两种身高父亲的儿子的身高，有向他们父辈的平均身高回归的趋势。对于这个一般结论的解释是:大自然具有一种约束力，使人类身高的分布相对稳定而不产生两极分化，这就是所谓的回归效应。

### 术语
- 特征
- 期望/标签
- 特征只有 1 个就是一元线性回归

#### 多元线性回归
我们这类房屋
准备数据

```python
data = np.genfromtxt('data/homeprices.csv',delimiter=',')
# print(data[])
x_data = data[:,:-1]
y_data = data[:,-1]
print(x_data)
print(y_data)
```
```
[2.6e+03 3.0e+00 2.0e+01]
 [3.0e+03 4.0e+00 1.5e+01]
 [3.2e+03 3.0e+00 1.8e+01]
 [3.6e+03 3.0e+00 3.0e+01]
 [4.0e+03 5.0e+00 8.0e+00]
 [4.1e+03 6.0e+00 8.0e+00]]
[550000. 565000. 610000. 595000. 760000. 810000.]
```



