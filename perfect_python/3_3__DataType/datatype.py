# coding:utf-8

def main():
    """
    データ型
    """

    # int
    print '>>> pythonのバージョンによって結果が異なる計算'
    print 3 / 2
    print 3.0 / 2.0

    # float
    print '>>> この加算はFalseとなる'
    print 0.1 + 0.1 + 0.1 == 0.3

    # complex
    print '>>> 複素数(実数 & 虚数)'
    print complex(1, 5)
    print 1 + 5j

    # mathモジュールのお勉強
    use_math_module()
    # decimalモジュールのお勉強
    use_decimal_module()

def use_math_module():
    """
    mathモジュールの利用
    """
    print '>>> mathモジュール'
    # 理解しやすくするためここでインポート
    import math

    print math.pi
    print math.cos(math.pi)
    print math.log(math.e)
    print math.sqrt(2)

def use_decimal_module():
    """
    decimalモジュールの利用
    """
    print '>>> decimalモジュール'
    from decimal import Decimal
    # 下記の結果はFalse(floatとdecimal)
    # Decimalは10進の数値を正確に扱える
    print 0.1 == Decimal('0.1')
    # つまり、こちらがTrue
    print Decimal('0.1') + Decimal('0.1') + Decimal('0.1') == Decimal('0.3')

if __name__ == "__main__":
    main()
