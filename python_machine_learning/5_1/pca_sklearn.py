#!/usr/bin/env python
# coding:utf-8

import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as po
import plotly.graph_objs as go
po.init_notebook_mode()

def load_wine():
    '''
    ワインデータセットをロードする
    '''
    # http://pythondatascience.plavox.info/scikit-learn/%E7%B7%9A%E5%BD%A2%E5%9B%9E%E5%B8%B0/
    # TODO: こっちのデータセットとの違いを調べる
    # $ wget http://pythondatascience.plavox.info/wp-content/uploads/2016/07/winequality-red.csv
#    path = r'winequality-red.csv'
#    df_wine = pd.read_csv(path, sep=';')
    path = 'wine.data'
    df_wine = pd.read_csv(path, header=None)
    print(df_wine.head)

    return df_wine

def plot_decision_regions(X, y, classifier, resolution=0.02):
    '''
    5.1.3 scikit-learnの主成分分析
    '''
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
    )

    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)

    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                marker=markers[idx], label=cl)


def run(X_train_std, X_test_std, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    # 主成分数を指定して、PCAのインスタンスを生成
    pca = PCA(n_components=2)
    # ロジスティック回帰のインスタンスを生成
    lr = LogisticRegression()
    # トレーニングんデータやテストデータをPCAに適合させる
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    # トレーニングデータをロジスティック回帰に適合させる
    lr.fit(X_train_pca, y_train)
    ## TODO: なんかグラフが逆になる。。。
    # 決定境界をプロット
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='lower left')
    plt.show()
    # 決定境界をプロット(テストデータセット)
    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='lower left')
    plt.show()

    # n_componentsをNoneに設定すると、すべての主成分が保持され、
    # 以下のようにして分散説明率にアクセスできる
    # (次元削減を実行する代わりに、すべての主成分がソートされた状態で返されるようにするため)
    pca2 = PCA(n_components=None)
    X_train_pca2 = pca2.fit_transform(X_train_std)
    # 分散説明率を計算
    print(pca2.explained_variance_ratio_)


def standardize(df):
    #------------------------------------------------------------------------------------
    # 主成分分析の最初の4ステップ
    #------------------------------------------------------------------------------------
    # 1. データを標準化
    # 2. 共分散行列を作成
    # 3. 共分散行列の固有ベクトルと固有値を求める
    # 4. 固有値を降順に並べ、固有ベクトルを順位付けする
    #------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------
    # 1. データを標準化
    #------------------------------------------------------------------------------------
    # 2列目以降のデータをXに、1列目のデータをyに格納
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    # 学習用 / テスト用データに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print('\nX(2列目以降のデータ) >>> \n%s' % X)
    print('\ny(1列目のデータ) >>> \n%s' % y)
    # 平均と標準偏差を用いて標準化(分散が1となるように)
    #------------------------------------------------------------------------
    # MEMO: なぜnormalizeでなくstandardizeなのか
    #------------------------------------------------------------------------
    # 以下の理由から、機械学習では標準化を用いるケースのほうが多いらしい
    # ・値が正規分布に従うようになるので重みの学習が行いやすくなる
    # ・normalizationより外れ値に対してロバストになる
    # normalize -> MinMaxScaler
    # standardize -> StandardScaler
    # 参考) http://qiita.com/gash717/items/eaba532730bad7a67efd
    #------------------------------------------------------------------------
    sc = StandardScaler()
    # 参考) normalize
    # sc = MinMaxScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test


if __name__ == '__main__':
    datasets = load_wine()
    X_train_std, X_test_std, y_train, y_test = standardize(datasets)
    run(X_train_std, X_test_std, y_train, y_test)
