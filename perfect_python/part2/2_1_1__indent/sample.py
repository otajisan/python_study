# coding:utf-8
def main():
    """
    インデントの練習
    """

    for i in range(1, 6):
        if i % 2 == 0:
            print("%sは偶数です。" % i)
        else:
            print("%sは奇数です。" % i)

# コマンドラインからの実行の場合
if __name__ == "__main__":
    main()
