#codign=utf-8

'''


'''

import numpy as np

class NaiveBayes:

    # fit 训练集
    def fit(self,X,y):
        n_samples, n_features = X.shape
        # 获取分类数量
        self._classes = np.unique(y)
        # 类别数
        n_classes = len(self._classes)
        # init mean var priors
        # 对于每一个类别每一个特征进行均值
        self._mean = np.zeros((n_classes,n_features),dtype=np.float64)
        # 
        self._var = np.zeros((n_classes,n_features),dtype=np.float64)
        # 这是一个 1 维向量
        self._priors = np.zeros(n_classes,dtype=np.float64)

        for c in self._classes:
            # 获取此类别训练集数据
            X_c = X[c==y]
            # 计算均值和标准差
            self._mean[c,:] = X_c.mean(axis=0)
            # 方差
            self._var[c,:] = X_c.var(axis=0)
            # 先验概率，频率
            self._priors[c] = X_c.shape[0] / float(n_samples)

    # 推测
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self,x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[dix])
            class_conditional = np.sum(np.log(self._pdf(idx,x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

    
    def _pdf(self,class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi * var)
        return numerator/ denominator