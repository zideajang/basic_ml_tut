# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# from IPython.display import Image 

def draw_likelihood(observations, mu, sigma):
    # 定义y轴取值
    plt.ylim(-0.02,1)
    # 定义一个画图范围
    x_locs =  np.linspace(-10, 10, 500)
    # 画出推断的概率分布的概率密度函数
    plt.plot(x_locs, stats.norm.pdf(x_locs, loc=mu, scale=sigma), label="inference")
    for obs in observations:
        plt.axvline(x=obs, ymin=0, ymax=stats.norm.pdf(obs, loc=mu, scale=sigma)+0.01, c="g")
    plt.axvline(x=obs, ymin=0, ymax=stats.norm.pdf(obs, loc=mu, scale=sigma)+0.01, c="g", label="probabilities")
    # 画出观测数据的概率
    plt.scatter(x=observations, y=[0 for _ in range(len(observations))], c="r", marker="o", label="obsevations")
    plt.legend()
    plt.grid()
    plt.title("mean={} sigma={}".format(str(mu), str(sigma)))
    plt.show()

if __name__ == "__main__":
    obs_mu, obs_sigma = 0, 4
    observations = np.random.normal(obs_mu, obs_sigma, 20)
    draw_likelihood(observations, mu=5, sigma=2)
    draw_likelihood(observations, mu=0, sigma=4)