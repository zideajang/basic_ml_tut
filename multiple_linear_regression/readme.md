$$ W = (X^TX)^{-1}X^TY$$

如果是两个参数和一个截断，也就是三个特征值，我们在等式两边除以矩阵是行不通，代替除以矩阵我们可以通过乘以逆矩阵来避免矩阵除法。值得注意的是只有方阵可逆。


```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```
读取数据
```
data = np.genfromtxt(r"data/data.csv",delimiter=',')
print(data)

```

```
[[100.    4.    9.3]
 [ 50.    3.    4.8]
 [100.    4.    8.9]
 [100.    2.    6.5]
```

切分数据为

```python
def equation_1():
    yy = regressor.predict(xx.reshape(xx.shape[0],1))

    plt.plot(xx,yy,label='degree=1')
    plt.axis([0, 28, 0, 28])
    plt.show()
```

![ml_031.png](https://upload-images.jianshu.io/upload_images/8207483-c83852685262cc6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在一元回归简单直线，由于模型过于简单无法表达曲线特征，这就是我们所说的欠拟合，可以同增加多项式来增强模型表达能力

```python
def equation_2():
    yy = regressor.predict(xx.reshape(xx.shape[0],1))
    plt.plot(xx,yy,label='degree=1')
    # 2 次项生成器
    quadratic_featurizer = PolynomialFeatures(degree=2)
    
    X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
    X_test_quadratic = quadratic_featurizer.transform(X_test)

    regressor_quadratic = LinearRegression()
    regressor_quadratic.fit(X_train_quadratic,y_train)

    plt.scatter(X_train, y_train)
    xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))
    yy_quadratic = regressor_quadratic.predict(xx_quadratic)
    plt1, = plt.plot(xx, yy, label="Degree1")
    plt2, = plt.plot(xx, yy_quadratic, label="Degree2")

    plt.axis([0, 28, 0, 28])
    
    # 0.8675443656345054
    print('Quadratic regression r-squared',regressor_quadratic.score(X_test_quadratic,y_test))

    plt.show()

```
![ml_032.png](https://upload-images.jianshu.io/upload_images/8207483-c89e4f3b8c7de409.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

显然 2 阶函数更好的拟合这些点相比简单线性回归，2 次线性回归提升从 0.81 提升到 0.87
```python
def equation_3():
    yy = regressor.predict(xx.reshape(xx.shape[0],1))
    plt.plot(xx,yy,label='degree=1')
    # 2 次项生成器
    biquadrate_featurizer = PolynomialFeatures(degree=4)
    
    X_train_biquadrate = biquadrate_featurizer.fit_transform(X_train)
    X_test_biquadrate = biquadrate_featurizer.transform(X_test)

    regressor_biquadrate = LinearRegression()
    regressor_biquadrate.fit(X_train_biquadrate,y_train)

    plt.scatter(X_train, y_train)
    xx_biquadrate = biquadrate_featurizer.transform(xx.reshape(xx.shape[0],1))
    yy_biquadrate = regressor_biquadrate.predict(xx_biquadrate)
    plt1, = plt.plot(xx, yy, label="Degree1")
    plt2, = plt.plot(xx, yy_biquadrate, label="Degree2")

    plt.axis([0, 28, 0, 28])
    # 0.8095880795782215
    print('Biquadrate regression r-squared',regressor_biquadrate.score(X_test_biquadrate,y_test))

    plt.show()

```

![ml_033.png](https://upload-images.jianshu.io/upload_images/8207483-2db01d313b287418.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

随着模型容量增加，当 4阶多项式的回归曲线会经过所有训练集点，但是我们通过观察会发现这个曲线虽然在训练集上表现优异，但是在测试集表现一眼看出他存在问题。这一次 4 阶方程又降低回了 0.80

这样就是我们所说的过拟合，而之前的一元线性回归，由于模型过于简单无法表达这些点，这种情况就是欠拟合