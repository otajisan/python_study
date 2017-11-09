'''
4.4 勾配
'''

import numpy as np
import matplotlib.pylab as plt

def multi_param_func(x):
    '''
    複数パラメータからなる関数
    '''
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    '''
    4.3.3 偏微分
    複数のパラメータからなる関数の微分
    '''
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    '''
    4.4.1 勾配法
    ここの例は勾配降下法
    (見方によって最大値を求める場合もあり、その場合は「勾配上昇法」を使うが、
    損失関数の符号に*-1すれば同じことなのであまり気にする必要はない)

    --- parameters ---
    f: 最適化したい関数
    lr: 学習率(learning rate)はハイパーパラメータ(手動で変えながら効果を確認する)
     * 1回の学習でどれだけ学習すべきか、どれだけパラメータを更新するか、を決めるのが学習率
    '''
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

if __name__ == '__main__':
    x = np.array([-3.0, 4.0])
    # 勾配の実装例
    grad = numerical_gradient(multi_param_func, x)
    print(grad)
    # 勾配硬化法
    lr = 10.0 # 学習率が大きすぎる場合
    lr = 1e-10 # 学習率が小さすぎる場合
    lr = 0.1 # 適切な学習率(この関数の場合の真の最小値(0,0)に限りなく近い結果([ -6.11110793e-10   8.14814391e-10])が出る)
    result = gradient_descent(multi_param_func, init_x=np.array(x), lr=lr, step_num=100)
    print(result)

