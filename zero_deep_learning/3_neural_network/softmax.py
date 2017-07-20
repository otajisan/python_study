# coding:utf-8
'''
3.5.1 恒等関数とソフトマックス関数
一般的に、出力層の活性化関数(σ(シグマ))は、
回帰 -> 恒等関数(入力をそのまま出力する関数)
分類 -> ソフトマックス関数
を使う

また、学習フェーズの出力層ではソフトマックス関数を利用するが、
推論フェーズの出力層ではソフトマックス関数を省略する
* 指数関数の計算量が多いため
* ニューラルネットワークのクラス分類では、
  出力の一番大きいニューロンに相当するクラスだけを認識結果とするが、
  ソフトマックス関数を利用しても、出力の一番大きいニューロンの位置は変わらないため
'''

import numpy as np
import matplotlib.pylab as plt

def softmax(a):
    '''
    単純なソフトマックス関数
    問題点として、オーバーフロー(eの1000乗など)に対応できない
    '''
    # e(ネイピア数、自然対数の底)のa乗を表す指数関数
    exp_a = np.exp(a)
    # 分子(入力信号aの指数関数)
    # 分母(すべての入力信号の指数関数の総和)
    return exp_a / np.sum(exp_a)


def softmax2(a):
    '''
    オーバーフロー対策版ソフトマックス関数
    定数C'を用いる
    '''
    # 定数C'は、一般的に入力値の最大の値を用いる
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)


def print_chart(x, y):
    '''
    結果をグラフ出力する
    '''
    plt.plot(x, y)
    # y軸の範囲指定
    plt.ylim(-0.1, 1.1)
    plt.show()


if __name__ == '__main__':
    # シンプルなソフトマックス
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print_chart(a, y)
    
    # オーバーフロー問題(nanになってしまう)
    a = np.array([1010, 1000, 990])
    print(softmax(a))

    # オーバーフロー対策(定数を使う)
    y = softmax2(a)
    print(y)
    print_chart(a, y)
    # そして、総出力の和は1になる(つまり、ソフトマックス関数は確率と同じ性質を持つ)
    print(np.sum(y))
