# coding:utf-8
'''
活性化関数の勉強
'''

import numpy as np
import matplotlib.pylab as plt

def sigmoid_function(x):
    '''
    シグモイド関数
    '''
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    print('### シグモイド関数')
    print('=== 入力x ===')
    x = np.arange(-5.0, 5.0 , 0.1)
    print(x)
    print('=== 出力y ===')
    y = sigmoid_function(x)
    print(y)
    plt.plot(x, y)
    # y軸の範囲指定
    plt.ylim(-0.1, 1.1)
    plt.show()

