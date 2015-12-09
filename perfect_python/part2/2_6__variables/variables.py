# coding:utf-8

def main():
    # 数値
    x = 100
    print x

    # 文字列
    mystr = 'hogehoge'
    print mystr

    # グローバル変数の定義関数を実行
    global_variables()
    print gstr

def global_variables():
    """
    グローバル変数のテスト
    """
    global gstr
    gstr = "global_fugafuga"

# コマンドラインからの実行の場合
if __name__ == "__main__":
    main()
