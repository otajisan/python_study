#!/usr/bin/env python
# coding:utf-8
from simple import MyClass

def call_myclass():
    '''
    クラスのメソッドを実行する
    '''
    obj = MyClass.MyClass('test')
    msg = obj.call()

    print msg

if __name__ == "__main__":
    print '### START main'
    call_myclass()
    print '### END main'
