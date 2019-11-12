# coding=utf-8
import matplotlib.pyplot as plt
import math
import numpy as np
def first_digital(x):
    while x >= 10:
        x /= 10
    return x

def demo_one():
    n = 1
    frequency = [0] * 9
    for i in range(1,1000):
        n *= i
        m = first_digital(n) -1
        frequency[m] += 1
    print frequency
    plt.plot(frequency, 'r-',linewidth=2)
    plt.plot(frequency,'go',markersize=8)
    plt.grid(True)
    plt.show()

def demo_two():
    x = np.arange(0.05,3,0.05)
    y1 = [math.log(a,1.5) for a in x]
    plt.plot(x,y1,linewidth=2,color='#007500',label='log1.5(x)')
    plt.plot([1,1],[y1[0],y1[-1]],"r--",linewidth=2)
    y2 = [math.log(a,2) for a in x]
    plt.plot(x,y2,linewidth=2,color='#9F35FF',label='log2(x)')
    y3 = [math.log(a,3) for a in x]
    plt.plot(x,y3,linewidth=2,color='#F75000',label='log3(x)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def run():
    demo_two()

if __name__ == "__main__":
    run()