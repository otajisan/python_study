# coding:utf-8

class MyClass:

    def __init__(self, msg):
        '''
        コンストラクタ
        '''
        self.msg = msg
        print '>>> constructor is called.'

    def __del__(self):
        '''
        デストラクタ
        '''
        print '>>> destructor is called.'

    def call(self):
        '''
        クラス/メソッドの練習
        '''
        msg = 'Simple Class is called. msg=[' + self.msg + ']'

        return msg
