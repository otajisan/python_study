# coding:utf-8

import numpy as np

def run_simple_perceptron(x1, x2, weight=0.5, bias=-0.7):
    '''
    重みとbias
    デフォルトはAND回路
    '''
    x = np.array([x1, x2])
    w = np.array([weight, weight])
    print(w*x)
    print(np.sum(w*x))
    print(float(np.sum(w*x) + bias))

    return 1 if float(np.sum(w*x) + bias) > 0 else 0

def AND(x1, x2):
    '''
    AND回路
    '''
    print('### AND ###')
    return run_simple_perceptron(x1, x2)


def NAND(x1, x2):
    '''
    NAND回路
    '''
    print('### NAND ###')
    w = -0.5
    b = 0.7
    return run_simple_perceptron(x1, x2, weight=w, bias=b)


def OR(x1, x2):
    '''
    OR回路
    '''
    print('### OR ###')
    w = 0.5
    b = -0.2
    return run_simple_perceptron(x1, x2, weight=w, bias=b)


def XOR(x1, x2):
    '''
    XOR回路(多層パーセプトロン)
    '''
    print('### XOR ###')
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    print("s1={s1} s2={s2} y={y}".format(s1=s1, s2=s2, y=y))
    return y


if __name__ == '__main__':
    input_data = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    print(">>>>>>>> AND回路")
    for data in input_data:
        print(AND(data[0], data[1]))

    print(">>>>>>>> NAND回路")
    for data in input_data:
        print(NAND(data[0], data[1]))

    print(">>>>>>>> OR回路")
    for data in input_data:
        print(OR(data[0], data[1]))
    # XOR回路
    # -> 非線形でしか分類できないので、一層のみの単純パーセプトロンではムリ！
    # なので、「多層パーセプトロン」で実現する
    # ここでは、AND x NAND x ORゲート(単純パーセプトロン)をつなげて
    # XOR回路(多層パーセプトロン)を実現する
    print(">>>>>>>> XOR回路")
    for data in input_data:
        print(XOR(data[0], data[1]))
