# coding:utf-8

def main():
    """
    リスト
    """

    x = [1, 2, 3.0, 'a', 'b', 'cc']

    print '>>> インデクシング(シーケンス番号を使ったアクセス)'
    print x[0]
    print x[1]
    print x[5]

    print '>>> スライス(listから指定した位置のサブリストを得る)'
    print x[1:2]
    print x[2:-1]
    print x[4:6]

    print '>>>>> こういうこともできる'
    print x[1:]
    print x[:3]
    print x[:]

    print '>>> イテレーション'
    print '>>>>> 全部'
    for item in x:
        print item

    print '>>>>> こんな利用方法もある'
    y = [1, 2, 3, 'abc']
    for c in y[3]:
        print c

    print '>>> リストの更新'
    print '>>>>> 元のリスト'
    a = ['hoge']
    print a
    print '>>>>> 要素の追加'
    a.append('fuga')
    print a
    print '>>>>> 要素の削除'
    a.remove('hoge')
    print a
    print '>>>>> インデックスを使った更新'
    a[0] = 'hoge'
    print a
    print '>>>>> スライスを使った更新'
    a[1:999] = ['foo', 'bar']
    print a

    print '>>> リストの反転'
    a.reverse()
    print a

    print '>>> リストのソート'
    b = [1, 0, 5, 99, 10]
    print b
    b.sort()
    print b

    # リスト内包表記
    list_comprehension()

def list_comprehension():
    """
    ループと条件を使って新しいlistを生成する特別なシンタックス
    """

    print '>>> リスト内包表記(リストコンプリヘンション)'
    print '>>>>> 0から9までの数字から偶数を取り出してリスト化'
    print [i for i in range(10) if i % 2 == 0]
    print [str(i) for i in range(10) if i % 2 == 0]

if __name__ == "__main__":
    main()
