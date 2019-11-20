# coding=utf-8
import numpy as np
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