#!/usr/bin/env python
# coding:utf-8

import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
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

    km = KMeans(
            n_clusters=3, # クラスタの個数
#            init='random', # セントロイドの初期値をランダムに選択
            init='k-means++', # k-means++法
            n_init=10, # 異なるセントロイドの初期値を用いたk-meansアルゴリズムの実行回数
            max_iter=300, # k-meansアルゴリズム内部の最大イテレーション回数
            tol=1e-04, # 収束と判定するための相対的な許容誤差
            random_state=0 # セントロイドの初期化に用いる乱数生成器の状態
        )
    # クラスタ中心の計算と各サンプルのインデックスの予測
    y_km = km.fit_predict(X)
    print(y_km)
    plt.scatter(
            X[y_km==0, 0],
            X[y_km==0, 1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
        )
    plt.scatter(
            X[y_km==1, 0],
            X[y_km==1, 1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
        )
    plt.scatter(
            X[y_km==2, 0],
            X[y_km==2, 1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3'
        )
    plt.scatter(
            km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids'
        )
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run()
