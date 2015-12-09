# coding:utf-8

def main():
    """
    タプル
    イミュータブル(不変)のシーケンス型
    """

    x = ('a', 'b', 'c', 'd', 'e')
    for i in x:
        print i

    print '>>> こちらはエラー(不変なので)'
    x[1] = 'f'

if __name__ == "__main__":
    main()
