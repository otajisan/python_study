# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def draw():
    '''
    2次関数を描画する
    '''
    a = 2
    b = 5
    x = np.arange(-3, 3, 0.1)
    y = a * x ** 2 + b
    plt.plot(x, y) 
    plt.show()

if __name__ == '__main__':
    draw()
