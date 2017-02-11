#!/usr/bin/env python
# coding:utf-8

import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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

def run(df):
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

    #------------------------------------------------------------------------------------
    # 2. 共分散行列を作成
    #------------------------------------------------------------------------------------
    # MEMO: 共分散行列
    # ベクトルの要素間の共分散の行列
    # スカラー値をとる確率変数における分散の概念を、多次元確率変数に拡張して行列としたものo
    # 分散共分散行列からは、データの相関を完全に失わせるような写像を作る変換行列を作ることができる
    # 統計学 -> 主成分分析(PCA)
    # 画像処理 -> カルーネン・レーベ変換(KL-transform)
    # として利用される(らしい)
    # np.cov -> 共分散行列の計算関数
    cov_mat = np.cov(X_train_std.T)

    #------------------------------------------------------------------------------------
    # 3. 共分散行列の固有ベクトルと固有値を求める
    #------------------------------------------------------------------------------------
    # 固有値と固有ベクトルを計算(eigen(アイジェン) = 固有)
    # np.linalg.eig -> 非対称正方行列を固有分解する関数
    # cf. np.linalg.eigh -> エルミート行列を分解する関数
    # 共分散行列などは対称行列となるため、エルミート行列のほうが数値的に安定する(らしい)
    # (と、言っているのになんでここでeigのほうを使っているのか・・・)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
#    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
    print('\nEigenvalues \n%s' % eigen_vals)

    ## ここでは、データセットを新しい特徴部分空間に圧縮する、という方法で次元削減する
    ## ここでのコツとして、データの大半の情報(分散)を含んでいる固有ベクトル(主成分)だけを選択する
    ## (つまり、高次元の複雑なデータのうち、重要なデータ部分を利用するという意味、と理解)

    #------------------------------------------------------------------------------------
    # MEMO: ここで、分散 = データの大半の情報、というイメージがががなので調べる
    #------------------------------------------------------------------------------------
    # 偏差；データの各値と平均値との差
    # 分散：偏差の2乗の平均値
    # 分散は平均値を中心とし、そこから各数値が(平均して)どのくらい散らばっているかを表す
    #   分散が大きい -> 平均から大きく離れているため、散らばりが大きい
    #   分散が小さい -> 平均に近いデータが多いため、散らばりが小さい
    # 
    # もう言葉見てもよく分からんので、↓やってグラフ化したほうがイメージつかみやすいわ
    #------------------------------------------------------------------------------------
    # 固有値を合計
    tot = sum(eigen_vals)
    # 分散説明率を計算
    # 分散説明率：固有値の合計に対する、固有値λ(スカラー)のj(特徴量)の割合
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    # 分散説明率の累積和を取得
    cum_var_exp = np.cumsum(var_exp)
    # グラフ表示
    create_variance_explanation_graph(var_exp, cum_var_exp)

    #------------------------------------------------------------------------------------
    # 5.1.2 特徴変換
    #------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------
    # 4. 固有値を降順に並べ、固有ベクトルを順位付けする
    #------------------------------------------------------------------------------------
    # (固有値, 固有ベクトル)のタプルのリストを作成
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    # 降順に並べる
    eigen_pairs.sort(reverse=True)

    #------------------------------------------------------------------------------------
    # PCA残りの2ステップ
    # 5. 上位k個の固有ベクトルから射影行列Wを作成する
    # 6. 射影行列Wを使ってd次元の入力データセットXを変換し、新しいk次元の特徴部分空間を取得する
    #------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------
    # 5. 上位k個の固有ベクトルから射影行列Wを作成する
    #------------------------------------------------------------------------------------
    # ここでは、最も大きい2つの固有値に対応する2つの固有ベクトルを集める
    # = 先程のグラフで見た結果から、分散の約60%を捉えることができる
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', w)
    # -> 13 x 2次元の射影行列Wが作成される

    print('>>> sampling')
    print(w) # 13 x 2次元(射影行列)
    print(X_train_std[0]) # 1 x 13次元(サンプリングしたベクトル)
    # 行列の内積を求める
    print(X_train_std[0].dot(w)) # 1 x 2次元(サンプリングしたベクトルをPCAの部分空間に変換)

    print('>>> こっちが本チャン(全データに適用)')
    # X_train_std -> 124 x 13次元
    # なので、X_train_pca -> 124 x 2次元
    X_train_pca = X_train_std.dot(w)
    print(X_train_pca)
    print(len(X_train_pca))

    # 2次元の散布図としてグラフにプロット
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    # 「クラスラベル」「点の色」「点の種類」の組み合わせからなるリストを生成してプロット
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m)
    plt.xlabel('PC 1 (主成分1)')
    plt.ylabel('PC 2 (主成分2)')
    plt.legend(loc='lower left')
    plt.show()


def create_variance_explanation_graph(var_exp, cum_var_exp):
    '''
    分散説明率の計算結果をグラフ化
    ポイント：
    1つ目の主成分だけで分散の40%近くを〆ている、という事実が分かる

    MEMO: ランダムフォレストとの違い
    ランダムフォレスト -> データの所属情報を使ってノードの不順度を計算する
    PCA -> 特徴量の軸に沿った値の散らばりを測定する
    '''
    # 分散説明率の棒グラフ
    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='個々の分散説明率')
    # 分散説明率の累積和の階段グラフ
    plt.step(range(1, 14), cum_var_exp, where='mid', label='分散説明率の累積和')
    plt.ylabel('Explained variance ratio(分散説明率)')
    plt.xlabel('Principal components(主成分)')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    datasets = load_wine()
    run(datasets)
