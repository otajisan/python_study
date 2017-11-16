'''
4.4.2 ニューラルネットワークに対する勾配
'''
import numpy as np

class simpleNet:
    def __init__(self):
        # 重みパラメータ
#        x = [
#            [0.47355232, 0.9977393, 0.84668094],
#            [0.85557411, 0.03563661, 0.69422093]
#        ]
#        self.W = np.array(x)
        self.W = np.random.randn(2, 3) # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = self.softmax(z)
        loss = self.cross_entropy_error(y, t)

        print('loss:', loss)
        return loss

#    def cross_entropy_error(self, y, t):
#        if y.ndim == 1:
#            t = t.reshape(1, t.size)
#            y = y.reshape(1, y.size)
#        
#        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
#        if t.size == y.size:
#            t = t.argmax(axis=1)
#             
#        batch_size = y.shape[0]
#        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

    def cross_entropy_error(self, y, t):
        '''
        交差エントロピー誤差
        '''
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

#    def softmax(self, x):
#        if x.ndim == 2:
#            x = x.T
#            x = x - np.max(x, axis=0)
#            y = np.exp(x) / np.sum(np.exp(x), axis=0)
#            return y.T 
#
#        x = x - np.max(x) # オーバーフロー対策
#        return np.exp(x) / np.sum(np.exp(x))

    def softmax(self, a):
        '''
        ソフトマックス関数
        '''
        # 定数C'は、一般的に入力値の最大の値を用いる
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a / np.sum(exp_a)

    def numerical_gradient(self, f, x):
        '''
        4.3.3 偏微分
        複数のパラメータからなる関数の微分
        '''
        h = 1e-4
        grad = np.zeros_like(x)

        for idx in range(x.shape[0]):

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

if __name__ == '__main__':
    net = simpleNet()
    print('### 入力データ')
    x = np.array([0.6, 0.9])
    print(x)

    print('### 入力データの結果を推定')
    p = net.predict(x)
    print(p)

    print('### 最大値のインデックス')
    print(np.argmax(p))

    print('### 正解ラベル')
    t = np.array([0, 0, 1])
    print(t)

    print('### 重み')
    print(net.W)

    f = lambda w: net.loss(x, t)
    # 以下と同義
    #def f(W):
    #    return net.loss(x, t)

    # 勾配を求める(偏微分)
    dW = net.numerical_gradient(f, net.W)
    print('### 勾配')
    print(dW)
    print('>>> この勾配の結果に基づいて、重みパラメータを更新していけばおｋ！')
