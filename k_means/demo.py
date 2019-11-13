# coding=utf-8
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def demo_one():
        # 创建数据集
        Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
                'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
        }
        # 格式化数据
        df = DataFrame(Data,columns=['x','y'])
        # print (df)
        kmeans = KMeans(n_clusters=3).fit(df)
        centroids = kmeans.cluster_centers_
        print(centroids)

        plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

        plt.show()

def demo_two():
        # 生产随机数
        np.random.seed(42)
        digits = load_digits()
        data = scale(digits.data)

        n_samples, n_features = data.shape
        n_digits = len(np.unique(digits.target))
        labels = digits.target
        sample_size = 300
def run():
        pass        

if __name__ == "__main__":
        run()