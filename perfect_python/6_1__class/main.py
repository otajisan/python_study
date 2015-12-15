#!/usr/bin/env python
# coding:utf-8
#import simple.MyClass
import simple.MyClass

def call_myclass():
    '''
    クラスのメソッドを実行する
    '''
    obj = simple.MyClass.MyClass('test')
    msg = obj.call()

    print msg

if __name__ == "__main__":
    call_myclass()
