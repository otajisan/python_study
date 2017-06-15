# coding:utf-8
'''
パーセプトロンのサンプル

パーセプトロンは
閾値 : θ
重み : w1, w2
入力信号 : x1, x2
出力 : y
'''

def run():
    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))

def AND(x1, x2):
    '''
    以下のようなANDゲートについて考える

    | x1 | x2 | y |
    |----|----|---|
    | 0  | 0  | 0 |
    | 1  | 0  | 0 |
    | 0  | 1  | 0 |
    | 1  | 1  | 1 |

    '''
    # ハイパーパラメータ定義
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

if __name__ == '__main__':
    run()
