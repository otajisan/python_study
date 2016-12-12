import numpy as np

class Perceptron(object):
    '''
    パーセプトロンによる分類器

    params:
    eta : float
        学習率(0.0 < eta <= 1.0)

    n_iter : int
        トレーニングデータのトレーニング回数(エポック数)

    properties:
    w_ : 1次元配列
        適合後の重み
    errors_ : リスト
        各エポックでの誤分類数
        ※エポック = データセットに対するトレーニングの最大回数
    '''
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        '''
        機械学習

        X : トレーニングデータ
        y : 目的関数
        '''
        # 重みw_をゼロベクトルに初期化
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        # トレーニング回数分反復
        for _ in range(self.n_iter):
            errors = 0
            # サンプル毎に重みを更新
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''
        総入力を計算
        '''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        '''
        1ステップ後のクラスラベルを返す
        '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
