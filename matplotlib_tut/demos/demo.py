# coding=utf-8
# 引入依赖
# numpy 在机器学习中，有一些背后算法，而且更多工作是调优，我们需要尽可能地观察机器学习过程
# 最好的方式就是可视化，因为我们对图形相对于文字更有感觉。易懂，我们在介绍 python 数据库可视化工具
# matplotlib 同时，也会介绍一些图形语言，我
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['path.simplify_threshold'] = 1.0

def create_figure_with_title():
    fig = plt.figure()  # 创建一个无坐标轴 figure
    fig.suptitle('Welcome to zidea world')  # 添加标题
    fig, ax_lst = plt.subplots(2, 2)  # 创建 figure with a 2x2 grid of Axes

# Matplotlib, pyplot and pylab: how are they related

'''
Matplotlib is the whole package and matplotlib.pyplot is a module in Matplotlib.

For functions in the pyplot module, there is always a "current" figure and axes 
(which is created automatically on request). 
在下面示例中先调用 plt.plot 创建坐标轴，然后调用 plt.plot 绘制曲线
plt.xlabel, plt.ylabel, plt.title 和 plt.legend set the axes labels and title
 and add a legend.
'''

def simple_plot():
    # 获取
    x = np.linspace(0, 2, 100)

    plt.plot(x, x, label='linear')
    plt.plot(x, x**2, label='quadratic')
    plt.plot(x, x**3, label='cubic')

    plt.xlabel('x label')
    plt.ylabel('y label')

    plt.title("Simple Plot")

    plt.legend()

def demo():
    y = np.random.rand(100000)
    y[50000:] *= 2
    y[np.logspace(1,np.log10(50000), 400).astype(int)] = -1
    mpl.rcParams['path.simplify'] = True

    mpl.rcParams['agg.path.chunksize'] = 0
    plt.plot(y)
    plt.show()

    mpl.rcParams['agg.path.chunksize'] = 10000
    plt.plot(y)

# Damped oscillation
def text_labelx_labely():
    # x 取值范围在 0 - 5 间
    x1 = np.linspace(0.0, 5.0, 100)
    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    # 设置显示窗口的大小
    fig, ax = plt.subplots(figsize=(5, 3))
    # 调整图距离边框左侧和底部的边距
    fig.subplots_adjust(bottom=0.15, left=0.2)
    # 绘制
    ax.plot(x1, y1)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Damped oscillation [V]')
'''
The x- and y-labels are automatically placed so that they clear the x- and 
y-ticklabels. Compare the plot below with that above, and note the y-label 
is to the left of the one above.
'''

'''
matplotlib.pyplot 提供用于定义和调整样式的方法 ，创建 figure, 
在 figure 中创建 plotting area , plots some lines in a plotting area, decorates 
the plot with labels, etc.

In matplotlib.pyplot various states are preserved across function calls, 
so that it keeps track of things like the current figure and plotting area, 
and the plotting functions are directed to the current axes 
(please note that "axes" here and in most places in the documentation refers to 
the axes part of a figure and not the strict mathematical term for more than one axis).

'''

'''
现在解释一下有关 y-axis 显示 1-4 而 x-axis 显示 0-3 原因
如果为 plot（）命令提供单个列表或数组，matplotlib 是一个 y 值序列，并自动为生成 x 值。
在 python 取值范围是 0 开始这个和其他语言没有任何区别，默认 x 向量的长度与 y 相同，这也是 x 数据是[0,1,2,3]。
'''

def draw_line():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('draw line')


def draw_line_2():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.ylabel('draw line 2')

'''
For every x, y pair of arguments, there is an optional third argument 
which is the format string that indicates the color and line type of the plot. 
he letters and symbols of the format string are from MATLAB, and you concatenate 
a color string with a line style string. The default format string is 'b-', 
which is a solid blue line. 
For example, to plot the above with red circles, you would issue
对于每个 x，y 参数对，都有一个可选的第三个参数，它是表示绘图颜色和线型的格式字符串。
格式字符串的字母和符号来自MATLAB，您可以将颜色字符串与线条样式字符串连接起来。
默认格式字符串为“b-”，这是一条蓝色实线。例如，要用红色圆圈绘制上述内容，您可以
'''

def draw_points():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
# D:\ml\basic_ml_tut\matplotlib_tut\screenshots
if __name__ == "__main__":
    draw_points()
    plt.show()