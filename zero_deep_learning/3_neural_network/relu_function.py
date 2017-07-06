# coding:utf-8
'''
活性化関数の勉強
'''

import numpy as np
import matplotlib.pylab as plt

def relu_function(x):
    '''
    ReLU関数
    '''
    return np.maximum(0, x)


if __name__ == '__main__':
    print('### ReLU関数')
    print('=== 入力x ===')
    x = np.arange(-5.0, 5.0 , 0.1)
    print(x)
    print('=== 出力y ===')
    y = relu_function(x)
    print(y)
    plt.plot(x, y)
    # y軸の範囲指定
    plt.ylim(-1.0, 5.0)
    plt.show()

