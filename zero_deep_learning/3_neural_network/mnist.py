'''
3.6 手書き数字認識
'''

import sys, os
import pickle
import numpy as np
import matplotlib.pylab as plt

sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from PIL import Image

def confirm_data():
    '''
    事前のデータ確認
    (どんなデータが入っているか)
    '''
    # normalize: 入力画像を0.0 - 1.0に正規化(ここではしない。ので、0-255のまま)
    # flatten: 入力画像を平ら(1次元配列)にする -> 1 x 28 x 28 = 784要素の1次元配列
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)

    # 試しに画像を表示
    img = x_train[0]
    label = t_train[0]
    print(label)
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    show_image(img)


def show_image(img):
    '''
    画像を表示
    '''
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def sigmoid(x):
    '''
    シグモイド関数
    '''
    return 1 / (1 + np.exp(-x))

def softmax(a):
    '''
    オーバーフロー対策版ソフトマックス関数
    定数C'を用いる
    '''
    # 定数C'は、一般的に入力値の最大の値を用いる
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)

def get_test_data():
    '''
    MNISTデータセットのうち、テストデータを返す
    '''
    # ここでは、normalize=Trueとしてデータを正規化(前処理：pre-processing)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test

def init_network():
    '''
    サンプルから重みとバイアスを読み込み
    '''
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    '''
    テストデータの結果を推論
    '''
    # 重みとバイアスの初期化
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 1層目
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # 2層目
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # 3層目
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def confirm_shape():
    '''
    3.6.3 バッチ処理
    ここでは各入力データと重みパラメータの形状を確認
    '''
    x, _ = get_test_data()
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    print('>>> 3.6.3 バッチ処理')
    print(x.shape)
    print(x[0].shape)
    print(W1.shape)
    print(W2.shape)
    print(W3.shape)

def print_chart(x, y):
    '''
    結果をグラフ出力する
    '''
    plt.plot(x, y)
    # y軸の範囲指定
    plt.ylim(-0.1, 1.1)
    plt.show()

if __name__ == '__main__':
    # MNISTデータセットの内容確認
    confirm_data()
    # 推論
    x, t = get_test_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

    # 3.6.3 バッチ処理
    confirm_shape()
    x, t = get_test_data()
    network = init_network()
    batch_size = 100
    accuracy_cnt = 0
    # 要は「100個ずつ処理する」ということ
    # まとめて処理(まとまった入力データのことをbatchと呼ぶ)することで、
    # 処理速度を圧倒的に上げることができる
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = predict(network, x_batch)
        # axis=1 -> 100x10の配列の中で、
        # 1次元目の要素ごとに最大値のインデックスを見つけることを指定
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i + batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
