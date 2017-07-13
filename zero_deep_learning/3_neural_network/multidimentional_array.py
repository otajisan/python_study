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
    # 1層目
    X = np.array([1.0, 0.5]) # 入力値
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 重み
    B1 = np.array([0.1, 0.2, 0.3]) # バイアス
    A1 = np.dot(X, W1) + B1 # 活性化関数(h)への入力値(a)
    Z1 = sigmoid(A1) # 活性化関数としてシグモイド関数を利用
    print(Z1)
    print_chart(A1, Z1)
    # 2層目
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(Z2)
    print_chart(A2, Z2)
    # 3層目
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)
    print(Y)
    print_chart(A3, Y)

def init_network():
    '''
    重みとバイアスの初期化
    '''
    network = {}
    # 1層目
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    # 2層目
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['B2'] = np.array([0.1, 0.2])
    # 3層目
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    '''
    入力から出力への伝達処理
    '''
    # 重み
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # バイアス
    b1, b2, b3 = network['B1'], network['B2'],network['B3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

def sigmoid(x):
    '''
    シグモイド関数
    '''
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    '''
    恒等関数(入力をそのまま出力する関数：σ(シグマ))
    '''
    return x

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
    # まとめ
    print('=== まとめ ===')
    network = init_network()
    x = np.array([1.0, 0.5]) # 入力値
    y = forward(network, x)
    print(y)
