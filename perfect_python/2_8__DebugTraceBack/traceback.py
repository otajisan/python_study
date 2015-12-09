# coding:utf-8

def main():
    mystr = "hoge"
    myint = 123

    # 文字列型と数値型の結合でエラーになる様子をトレースバックで確認できるお！
    concat(mystr, myint)

def concat(a, b):
    """
    こんなのが出るよ
    Traceback (most recent call last):
      File "traceback.py", line 14, in <module>
        main()
      File "traceback.py", line 8, in main
        concat(mystr, myint)
      File "traceback.py", line 11, in concat
        return a + b
    TypeError: cannot concatenate 'str' and 'int' objects
    """
    return a + b

if __name__ == "__main__":
    main()
