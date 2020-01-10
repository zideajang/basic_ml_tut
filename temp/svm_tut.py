# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC # "Support vector classifier"
'''
通过 scipy 提供方法来创建一些线性可分数据用于训练 SVM
'''
X, y = make_blobs(n_samples = 100, centers = 2, random_state = 0, cluster_std = 0.50)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'summer')

'''
根据我们所了解 SVM 可以做判别分类器。 通过简单地在二维或多维将一个类别从另一个类别区分开。
'''
xfit = np.linspace(-1, 3.5)
# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'summer')
# 绘制可能分割线
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
   yfit = m * xfit + b
   plt.plot(xfit, yfit, '-k')
   plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
   color = '#AAAAAA', alpha = 0.4)
plt.xlim(-1, 3.5)


'''
从上面图, 很容易看到间距"margins" 来将两个类别清晰分离 SVM 通过算法找到具有最大间隔。
今天通过使用 Scikit-Learn’ 提供的 SVC 来使用数据训练我们 SVM 模型。今天我们使用的是线性核
'''

model = SVC(kernel = 'linear', C = 1E10)
model.fit(X, y)

'''
输出如下
SVC(C=10000000000.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
kernel='linear', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False)
现在, for a better understanding, the following will plot the decision functions for 2D SVC −
'''

def decision_function(model, ax = None, plot_support = True):
   if ax is None:
      ax = plt.gca()
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()
   '''
   对于评估模型, 
   '''
   x = np.linspace(xlim[0], xlim[1], 30)
   y = np.linspace(ylim[0], ylim[1], 30)
   Y, X = np.meshgrid(y, x)
   xy = np.vstack([X.ravel(), Y.ravel()]).T
   P = model.decision_function(xy).reshape(X.shape)
   '''
   Next, we need to plot decision boundaries and margins as follows −
   '''
   ax.contour(X, Y, P, colors = 'k', levels = [-1, 0, 1], 
   alpha = 0.5, linestyles = ['--', '-', '--'])
   '''
   Now, similarly plot the support vectors as follows −
   '''
   if plot_support:
      ax.scatter(model.support_vectors_[:, 0],
      model.support_vectors_[:, 1], s = 300, linewidth = 1, facecolors = 'none')
   ax.set_xlim(xlim)
   ax.set_ylim(ylim)
'''
现在, 时n to fit our models as follows −
'''
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'summer')
decision_function(model)
plt.show()