# coding=utf-8
import matplotlib.pyplot as plt

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

def run():
    demo_one()
    

if __name__ == "__main__":
    run()