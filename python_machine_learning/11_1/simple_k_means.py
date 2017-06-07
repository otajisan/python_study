#!/usr/bin/env python
# coding:utf-8

import matplotlib.pyplot as plt


from sklearn.datasets import make_blobs

def run():
    X, y = make_blobs(
            n_samples=150, # サンプル点の総数
            n_features=2, # 特徴量の個数
            centers=3, # クラスタの個数
            cluster_std=0.5, # クラスタ内の標準偏差
            shuffle=True, # サンプルをシャッフル
            random_state=0 # 乱数発生器の状態を指定
          )
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', s=50)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run()
