# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import math

def draw_equation():
    x = np.linspace(-5,5,100)
    y = x*x
    plt.plot(x, y, '-r', label='y=2x+1')
    plt.title('Graph of $y=x^2$')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
def draw_equation_two():
    arr = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5],np.int64)
    print arr.shape
    fig, ax = plt.subplots()
    
    for i in range(arr.shape[0]):
        # print int(arr[i])
        x = np.linspace(-10,10,100)

        y  = arr[i] - x
        plt.plot(x, y, '-r', label='$x_1 + x_2 = $' + str(arr[i]))

    circle1 = plt.Circle((0, 0), math.sqrt(2), color='blue',fill=False)
    ax.add_artist(circle1)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title('Graph of $y=x^2$')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    x_major_locator=plt.MultipleLocator(1)
    y_major_locator=plt.MultipleLocator(1)
    
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_major_locator(x_major_locator)

    plt.xlim(-5,5)
    plt.ylim(-5,5)

    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)
    plt.grid()
    plt.show()

def main():
    draw_equation_two()
if __name__ == '__main__':
    # graph('x*3+2*x', range(-10, 11))
    main()
    