# coding:utf-8

import numpy as np

def run_perceptron(x1, x2, weight=0.5, bias=-0.7):
    '''
    重みとbias
    '''
    print('### run perceptron ###')
    x = np.array([x1, x2])
    w = np.array([weight, weight])
    print(w*x)
    print(np.sum(w*x))
    print(float(np.sum(w*x) + bias))

    return 1 if float(np.sum(w*x) + bias) > 0 else 0


if __name__ == '__main__':
    input_data = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    print(">>>>>>>> AND回路")
    for data in input_data:
        print(run_perceptron(data[0], data[1]))

    print(">>>>>>>> NAND回路")
    w = -0.5
    b = 0.7
    for data in input_data:
        print(run_perceptron(data[0], data[1], weight=w, bias=b))

    print(">>>>>>>> OR回路")
    w = 0.5
    b = -0.2
    for data in input_data:
        print(run_perceptron(data[0], data[1], weight=w, bias=b))
    # XOR回路
    # -> 非線形でしか分類できないので、単純パーセプトロンではムリ！
