# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def draw():
    '''
    １次関数を描画する
    '''
    a = 1 # 傾き
    b = 1 # 切片
    x = np.arange(-3, 3, 0.1)
    y = a * x + b
    plt.plot(x, y) 
    plt.show()

if __name__ == '__main__':
    draw()
