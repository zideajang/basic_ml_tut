# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 训练集数据
X_train = [[6],[8],[10],[14],[18]]
# 训练标签
y_train = [[7],[9],[13],[17.5],[18]]

# 测试数据集合测试标签
X_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

regressor = LinearRegression()
regressor.fit(X_train,y_train)
# 指定开始值、终止值和指定点数量
xx = np.linspace(0,26,100)
plt.scatter(X_train,y_train)


def equation_1():
    yy = regressor.predict(xx.reshape(xx.shape[0],1))

    plt.plot(xx,yy,label='degree=1')
    plt.axis([0, 28, 0, 28])
    plt.show()

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



if __name__ == "__main__":
    equation_3()
    

