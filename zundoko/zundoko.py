# coding:utf-8

from random import choice

ZUNDOKO_MESSAGE = 'ズンズンズンズンドコ'
KIYOSHI = 'キ・ヨ・シ!'

def zundoko():
    check_list = []
    while ''.join(check_list[-5:]) != ZUNDOKO_MESSAGE:
        check_list.append(choice(['ズン', 'ドコ']))
        print check_list[-1]
    print KIYOSHI

if __name__ == '__main__':
    zundoko()
