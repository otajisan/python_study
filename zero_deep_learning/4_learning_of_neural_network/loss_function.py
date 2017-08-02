'''
4.2 損失関数

【損失関数】
ニューラルネットワークの性能の悪さを示す指標
* 2乗誤差
* 交差エントロピー誤差
'''

import numpy as np


def mean_squared_error(y, t):
    '''
    2乗誤差
    '''
    print('>> by mean_squared_error')
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    '''
    交差エントロピー誤差
    '''
    print('>> by cross_entropy_error')
    delta = 1e-7 
    return -np.sum(t * np.log(y + delta))


if __name__ == '__main__':
    # 3.6章の手書き数字認識
    # MEMO: 正解ラベルを1、それ以外を0とする表記法を「one-hot表現」という
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 教師データ(「2」が正解、というデータ)
    # ソフトマックス関数の出力(2の確率が最も高い)
    print('### 2の確率が最も高い出力データ ###')
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(mean_squared_error(np.array(y), np.array(t)))
    print(cross_entropy_error(np.array(y), np.array(t)))

    # 7の確率が最も高い
    print('### 7の確率が最も高い出力データ ###')
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0] # 7の確率が最も高い
    print(mean_squared_error(np.array(y), np.array(t)))
    print(cross_entropy_error(np.array(y), np.array(t)))
