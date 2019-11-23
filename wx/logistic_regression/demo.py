# coding=utf-8
import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import Image 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from plotly.offline import donwload

def logit(p):
    return np.log(p/(1-p))


def draw_logistic_function():
    p = np.arange(1e-4, 1, 0.0001)
    y = logit(p)
    print(y)
    plt.plot(p, y , color='red', lw=2)
    plt.axvline(x=0, lw=3, label='undefined')
    plt.axvline(x=1, lw=2, label='undefined')
    plt.xticks([i/100 for i in range(0, 125, 25)])
    plt.grid()
    plt.legend()

def draw_binary_linear_equation():
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
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def draw_sigmoid_equation():
    z = np.arange(-5, 5, 0.001)
    y = [sigmoid(i) for i in z]
    plt.plot(z, y, color='red', lw=2)
    plt.axhline(y=0, lw=2, label="undefined")
    plt.axhline(y=1, lw=2, label="undefined")
    plt.grid()
    plt.legend()
if __name__ == "__main__":
    draw_sigmoid_equation()
    plt.show()
    