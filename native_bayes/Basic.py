import numpy as np

class NaiveBayes:
    def __init__(self):
        # 记录训练集的变量
        self._x = self._y = None
        # 
        # _func 模型核心,根据输入 x 和 y 输出对应的后验概率
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dic = self._feat_dics = None

    def __getitem__(self, item):
        if isinstance(item,str):
            return getattr(self,"_" + item)
    
    def predict_one(self, x, get_raw_result=False):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = x[:]
        x = self._transfer_x(x)