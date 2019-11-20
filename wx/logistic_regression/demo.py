# coding=utf-8
import numpy as np
<<<<<<< HEAD
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import Image 
# from plotly.offline import donwload

# 定义参数
theta_1 = 3
theta_2 = 2
bias = 0.8
# 获取网格
x_1 = y_1 = np.arange(-5, 5, 0.1)
x_1, y_1 = np.meshgrid(x_1, y_1)
# 进行二元线性回归模型计算
y_hat = \
(theta_1 * np.ravel(x_1) + 
 theta_2 * np.ravel(y_1) + bias).reshape(x_1.shape)

data = [go.Surface(z=y_hat)]
layout = go.Layout(title="Linear regression")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
=======
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

x = np.linspace(0,4,50)
y = func(x, 2.5, 1.3, 0.5)
yn = y + 0.2*np.random.normal(size=len(x))

popt, pcov = curve_fit(func, x, yn)

plt.figure()
plt.plot(x, yn, 'ko', label="Original Noised Data")
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()
>>>>>>> 745db95d4c1f57a0dce91f68928cacc4011d2abd
