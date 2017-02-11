#!/usr/bin/env python
# coding:utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def load_wine():
    '''
    ワインデータセットをロードする
    '''
    # http://pythondatascience.plavox.info/scikit-learn/%E7%B7%9A%E5%BD%A2%E5%9B%9E%E5%B8%B0/
    path = 'wine.data'
    df_wine = pd.read_csv(path, header=None)
    print(df_wine.head)

    return df_wine


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    '''
    こちらの記事より
    http://qiita.com/gash717/items/5ad68ed192f802c6ad36
    '''
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # 最小値, 最大値からエリアの領域を割り出す
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # resolutionの間隔で区切った領域を定義
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
            alpha=1.0, linewidth=1, marker='o',
            s=55, label='test set')


def run():

    #------------------------------------------------------------------------
    # d次元のデータセットを標準化する(dは特徴量の個数)
    # (PCAの節と同じのため省略)
    #------------------------------------------------------------------------
    df = load_wine()
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

    #------------------------------------------------------------------------
    # ここからLDA
    #------------------------------------------------------------------------
    # 次元数(ここでは2)を指定して、LDAのインスタンスを生成
    lda = LDA(n_components=2)
    # LDAによってトレーニングデータセットを低次元のデータに変換
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    #------------------------------------------------------------------------
    # 変換されたデータをロジスティック回帰で分類
    # (トレーニングデータを分類)
    #------------------------------------------------------------------------
    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    # 結果を見ると、1個誤分類されている
    plt.show()

    #------------------------------------------------------------------------
    # テストデータを分類
    # (トレーニングデータをロジスティック回帰で分類したところ、
    # 1つ誤分類されたが、同じくLDAで2次元に変換したデータをロジスティック回帰で
    # 分類するとどうなるか？という検証)
    #------------------------------------------------------------------------
    # テストデータもLDAで低次元のデータに変換
    X_test_lda = lda.transform(X_test_std)
    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    # 結果を見るときれいに分類されているのでこれでおｋ！ってことでいい？
    plt.show()

    #------------------------------------------------------------------------
    # おまけ : 正則化のおさらい(3.3.4 正則化による過学習への対処)
    #------------------------------------------------------------------------
    # トレーニングデータで1個誤分類が出たのをうまいこと分類したい
    # LDAのほうでなく、ロジスティック回帰のパラメータを変更してみる
    #
    # 正則化の強さを下げる(決定境界をずらせる)
    # 正則化パラメータ -> λ
    # scikit-learnのロジスティック回帰に実装されているパラメータ -> C (C = 1 / λ)
    # なので、λの値を小さくする(Cの値を大きくする)ことで、正則化が弱まるはず
    #
    # Cはデフォルト1なので、1より大きい値だとうまく分類できるようになる？

    # ここではデフォルトのL2正則化
    # Cの値を適当に変える
    # default: C=1.0, penalty='l2'
    # 参考) sklearnのロジスティック回帰のデフォルト値
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    lr = LogisticRegression(C=10)
#    lr = LogisticRegression(C=0.3, penalty='l1')
    lr = lr.fit(X_train_lda, y_train)
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.show()

    # もう少しマジメにやるならクロスバリデーションで
    tuned_parameters = [
        {
            'C': [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20],
#            'penalty': ['l1', 'l2']
        }
    ]

    clf = GridSearchCV(
            LogisticRegression(),
            tuned_parameters, # 検査対称パラメータ
            cv=5, # クロスバリデーションの回数
            scoring='accuracy', #
        )
    clf.fit(X_train_lda, y_train)
    print("##### 各試行結果 #####")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print("##### ベストパラメータ #####")
    print(clf.best_estimator_)
    y_true, y_pred = y_train, clf.predict(X_train_lda)
    print(classification_report(y_true, y_pred))
    plot_decision_regions(X_train_lda, y_train, classifier=clf)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    # この訓練データの場合、
    # L2正則化の場合 -> C=2くらい
    # L1正則化の場合 -> C=0.3くらい
    # が適正っぽい
    plt.show()


if __name__ == '__main__':
    run()
