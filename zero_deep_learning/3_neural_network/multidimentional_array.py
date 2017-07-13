# coding:utf-8
'''
NumPyを使った多次元配列の扱い方
'''

import numpy as np
import matplotlib.pylab as plt

def run_simple():
    '''
    単純な行列
    '''
    # 1次元配列
    A = np.array([1, 2, 3, 4])
    print(np.ndim(A))
    print(A.shape)
    print(A.shape[0])

    # 3 x 2の配列
    B = np.array([
        [1, 2],
        [3, 4],
        [5, 6,],
    ])
    print(np.ndim(B))
    print(B.shape)

def run_dot():
    '''
    行列の内積計算
    '''
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    # 1 x 5 + 2 x 7 = 19 ...
    print(np.dot(A, B))

    C = np.array([[1, 2, 3], [4, 5, 6]]) # 2 x 3の行列
    D = np.array([[1, 2], [3, 4], [5, 6]]) # 3 x 2の行列
    # 1 x 1 + 2 x 3 + 3 x 5 = 22 ...
    print(np.dot(C, D))

def run_neural_network_dot_product():
    '''
    3.3.3 ニューラルネットワークの内積
    '''
    X = np.array([1, 2])
    W = np.array([[1, 3, 5], [2, 4, 6]])
    Y = np.dot(X, W)
    print(Y)

    # 3.4 3層のニューラルネットワークの実装
    print('=== 3層のニューラルネットワークの実装 ====')
    X = np.array([1.0, 0.5]) # 入力値
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 重み
    B1 = np.array([0.1, 0.2, 0.3]) # バイアス
    A1 = np.dot(X, W1) + B1 # 活性化関数(h)への入力値(a)
    Z1 = sigmoid(A1) # 活性化関数としてシグモイド関数を利用
    print(Z1)
    print_chart(A1, Z1)

def sigmoid(x):
    '''
    シグモイド関数
    '''
    return 1 / (1 + np.exp(-x))

def print_chart(x, y):
    '''
    結果をグラフ出力する
    '''
    plt.plot(x, y)
    # y軸の範囲指定
    plt.ylim(-0.1, 1.1)
    plt.show()

if __name__ == '__main__':
    run_simple()
    run_dot()
    run_neural_network_dot_product()
