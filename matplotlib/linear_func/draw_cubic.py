# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def draw():
    '''
    3次関数を描画する
    '''
    x = np.arange(-3, 3, 0.1)
    y = x ** 3
    plt.plot(x, y) 
    plt.show()

if __name__ == '__main__':
    draw()
