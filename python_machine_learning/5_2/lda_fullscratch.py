#!/usr/bin/env python
# coding:utf-8

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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


def standardize(df):
    #------------------------------------------------------------------------------------
    # 1. データを標準化
    #------------------------------------------------------------------------------------
    # 2列目以降のデータをXに、1列目のデータをyに格納
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    # 学習用 / テスト用データに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print('\nX(2列目以降のデータ) >>> \n%s' % X)
    print('\ny(1列目のデータ) >>> \n%s' % y)
    sc = StandardScaler()
    # 参考) normalize
    # sc = MinMaxScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test

def run():
    '''
    LDA(線形判別分析)のサンプル
    '''
    #------------------------------------------------------------------------
    # LDAの流れ
    # 1. d次元のデータセットを標準化する(dは特徴量の個数)
    # 2. クラスごとにd次元の平均ベクトルを計算する
    # 3. クラス間変動行列(SB)と、クラス内変動行列(SW)を生成する
    # 4. 行列(SW^-1SB)の固有ベクトルと対応する固有値を計算する
    # 5. d x k次元の変換行列Wを生成するために、最も大きいk個の固有ベクトルを選択する
    # 6. 変換行列Wを使ってサンプルを新しい特徴空間へ射影する
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    # 1. d次元のデータセットを標準化する(dは特徴量の個数)
    # (PCAの節と同じのため省略)
    #------------------------------------------------------------------------
    X_train_std, X_test_std, y_train, y_test = standardize(load_wine())

    #------------------------------------------------------------------------
    # 2. クラスごとにd次元の平均ベクトルを計算する(本書5.2.1)
    #------------------------------------------------------------------------
    print('>>> 2. クラスごとにd次元の平均ベクトルを計算する(本書5.2.1)\n')
    # 有効桁数4桁で丸め
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in range(1, 4):
        # np.meanで平均(平均ベクトル)を計算
        mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
        print('MV %s: %s\n' % (label, mean_vecs[label - 1]))

    #------------------------------------------------------------------------
    # 3. クラス間変動行列(SB)と、クラス内変動行列(SW)を生成する
    #------------------------------------------------------------------------
    # 特徴量の個数
    d = 13
    # クラス内変動行列を計算
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d))
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)
            # 変動行列(Si)
            class_scatter += (row - mv).dot((row - mv).T)
            # 変動行列を加算し、クラス内変動行列を求める
            S_W += class_scatter
    print('クラス内変動行列: %sx%s' % (S_W.shape[0], S_W.shape[1]))
    print('Class label distribution(クラスラベルの個数): %s' % np.bincount(y_train)[1:])

    #------------------------------------------------------------------------
    # 4. 行列(SW^-1SB)の固有ベクトルと対応する固有値を計算する
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    # 5. d x k次元の変換行列Wを生成するために、最も大きいk個の固有ベクトルを選択する
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    # 6. 変換行列Wを使ってサンプルを新しい特徴空間へ射影する
    #------------------------------------------------------------------------


if __name__ == '__main__':
    run()
