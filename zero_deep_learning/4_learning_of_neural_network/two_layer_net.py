import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt

class TwoLayerNet:
    '''
    4.5.1 2層ニューラルネットワークのクラス
    '''

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        '''
        パラメータの初期化を行う
        input_size: 入力層のニューロンの数
        hidden_size: 隠れ層のニューロンの数
        output_size: 出力層のニューロンの数

        例：MNIST問題の場合
        input_size=784
        output_size=10
        hidden_size: 適当な値
        '''
        # 重みの初期化
        self.params = {} # パラメータ保持用のインスタンス変数(W1, b1: 1層目の重み, バイアス)
        self.params['W1'] = weight_init_std * \
                np.random.randn(input_size, hidden_size) # ガウス分布に従う乱数
        self.params['b1'] = np.zeros(hidden_size) # ゼロ
        self.params['W2'] = weight_init_std * \
                np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        '''
        推論を行う
        x: 画像データ
        '''
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)

        return y

    def sigmoid(self, x):
        '''
        シグモイド関数
        '''
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        '''
        MEMO: 5章でやる
        '''
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

    def softmax(self, x):
        '''
        ソフトマックス関数
        '''
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x) # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))

    def loss(self, x, t):
        '''
        損失関数のラッパー
        x: 入力データ
        t: 教師データ

        MEMO: この損失関数を減らす重みパラメータを勾配を使って求める
        '''
        y = self.predict(x)
        return self.cross_entropy_error(y, t)

    def cross_entropy_error(self, y, t):
        '''
        損失関数(交差エントロピー誤差)
        '''
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)
             
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

    def accuracy(self, x, t):
        '''
        認識精度を求める
        '''
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, f, x):
        '''
        勾配を求める偏微分関数
        '''
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)
    
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x) # f(x+h)
        
            x[idx] = tmp_val - h 
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)
        
            x[idx] = tmp_val # 値を元に戻す
            it.iternext()   
        
        return grad

    def gradient(self, x, t):
        '''
        5章でやる高速版
        '''
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = self.sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def numerical_gradient_wrapper(self, x, t):
        '''
        パラメータに対する勾配を求める
        x: 入力データ
        t: 教師データ
        grads: 勾配(W1, b1: 1層目の重みの勾配, バイアスの勾配)
        '''
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = self.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self.numerical_gradient(loss_W, self.params['b2'])

        return grads

    def show_graph(self):
        '''
        グラフを描画する
        '''
        # グラフの描画
        x = np.arange(len(train_acc_list))
        plt.plot(x, train_acc_list, label='train acc')
        plt.plot(x, test_acc_list, label='test acc', linestyle='--')
        plt.plot(x, train_loss_list, label='train loss')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

if __name__ == '__main__':
    '''
    4.5.2 ミニバッチ学習の実装

    【ここでの処理を日本語にすると・・・】
    ミニバッチのサイズを100とし、
    毎回60,000個のMNIST訓練データからランダムに100個のデータを抜き出す。
    100個のミニバッチを対象に勾配を求め、確率勾配降下法(SGD)でパラメータを更新する。
    勾配法によるパラメータ更新の回数は10,000回とする。
    '''
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # ハイパーパラメータ
    iters_num = 10000
    #iters_num = 1000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    # 1エポックあたりの繰り返し数
    # 1エポック：学習において訓練データをすべて使いきったときの回数に対応
    # e.g) 10,000個の訓練データに対して100個のミニバッチで学習する場合、SGDを100回繰り返したら
    #      100 x 100 = 10,000 = すべての訓練データを見た
    #      となるので、この場合1エポック = 100回となる
    iter_per_epoch = max(train_size / batch_size, 1)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    print('parameters, train_size={train_size}, batch_size={batch_size}'.format(
            train_size=train_size,
            batch_size=batch_size
        ))

    # SGDによるパラメータ更新
    for i in range(iters_num):
        if i % 1000 == 0:
            print("勾配法による {i} 回目の更新".format(i=i))
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        #grad = network.numerical_gradient_wrapper(x_batch, t_batch)
        # 高速版
        grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)

        # 1エポックごとに認識精度を計算
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            train_loss_list.append(loss)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    network.show_graph()
