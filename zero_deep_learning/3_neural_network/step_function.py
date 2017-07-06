# coding:utf-8
'''
活性化関数の勉強
'''

import numpy as np
import matplotlib.pylab as plt

def simple_step_function(x):
    '''
    単純なステップ関数
    '''
    if x > 0:
        return 1
    return 0


def step_function_for_matrix(x):
    '''
    NumPy配列対応のステップ関数
    '''
    return np.array(x > 0, dtype=np.int)


def run_sample_numpy():
    '''
    上記関数のイメージを掴みやすくするためのNumPyのサンプル
    '''
    x = np.array([-1.0, 1.0 , 2.0])
    print(x)
    # 行列内のすべての値について評価できちゃう！
    y = x > 0
    print(y)
    # 我々の望むステップ関数はbooleanでなくint型で欲しい！
    # ので、boolean -> int変換
    y = y.astype(np.int)
    print(y)


if __name__ == '__main__':
    print('### 単純なステップ関数')
    print(simple_step_function(3.0))

    print('### NumPyを使ったサンプル')
    run_sample_numpy()

    print('### NumPy利用ステップ関数')
    print('=== 入力x ===')
    x = np.arange(-5.0, 5.0 , 0.1)
    print(x)
    print('=== 出力y ===')
    y = step_function_for_matrix(x)
    print(y)
    plt.plot(x, y)
    # y軸の範囲指定
    plt.ylim(-0.1, 1.1)
    plt.show()

