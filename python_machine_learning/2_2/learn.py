# coding:utf-8

def load_iris():
    '''
    あやめデータの読み込み
    '''
    import pandas as pd
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    res = df.tail()
    print(res)
    # 145  6.7  3.0  5.2  2.3  Iris-virginica
    # 146  6.3  2.5  5.0  1.9  Iris-virginica
    # 147  6.5  3.0  5.2  2.0  Iris-virginica
    # 148  6.2  3.4  5.4  2.3  Iris-virginica
    # 149  5.9  3.0  5.1  1.8  Iris-virginica
    return df


def create_label(df):
    '''
    あやめデータのクラスラベルを生成
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    # 1 - 100行目の目的変数の抽出
    y = df.iloc[0:100, 4].values
    # Iris-stosa -> -1, Iris-virginica -> 1 とする
    y = np.where(y == 'Iris-setosa', -1, 1)
    # 1 - 100行目の1, 3列目を抽出
    X = df.iloc[0:100, [0, 2]].values
    # 品種setosaのプロット
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # 軸のラベルの設定
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    # 凡例
    plt.legend(loc='upper left')
    # 出力
    plt.show()

    return [X, y]


def create_perceptron_obj():
    '''
    パーセプトロンのインスタンスを生成
    '''
    import perceptron
    eta = 0.1
    n_iter = 10
    return perceptron.Perceptron(eta, n_iter)


def run_perceptron(X, y, ppn):
    '''
    パーセプトロンによる分類
    '''
    import matplotlib.pyplot as plt
    # 学習
    ppn.fit(X, y)
    # プロット
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    from matplotlib.colors import ListedColormap
    import numpy as np
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o','^', 'v')
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
    # 予測結果をもとのグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


if __name__ == '__main__':
    df = load_iris()
    [X, y] = create_label(df)

    ppn = create_perceptron_obj()
    run_perceptron(X, y, ppn=ppn)

    # 決定境界のプロット
    import matplotlib.pyplot as plt
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
