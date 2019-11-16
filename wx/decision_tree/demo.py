# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    p = np.arange(0.001,1,0.001,dtype=np.float)
    gini = 2 * p * (1-p)
    h = -(p* np.log2(p) + (1-p) * np.log2(1-p)) /2
    err = 1 - npπ