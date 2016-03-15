# coding:utf-8

from random import choice

class ZundokoKiyoshi:

    CORRECT_MESSAGE = 'ズンズンズンズンドコ'

    RESULT_MESSAGE = 'キ・ヨ・シ!'

    _message_list = ['ズン', 'ドコ']

    _result = []

    def say_rand(self):
        '''
        メッセージをランダムに返す
        '''
        return choice(self._message_list)

    def append(self, message):
        '''
        リストにメッセージを追加する
        '''
        self._result.append(message)

    def empty(self):
        '''
        リストを空にする
        '''
        self._result = []

    def is_correct(self):
        '''
        リスト内のメッセージを確認し、メッセージが正しいかどうかを返す
        '''
        return ''.join(self._result[-5:]) == self.CORRECT_MESSAGE

    def print_result(self):
        '''
        結果メッセージを返す
        '''
        print ''.join(self._result) + self.RESULT_MESSAGE

    def check(self):
        '''
        ズンドコキヨシを実行する
        '''
        while (self.is_correct() is False):
            self.append(self.say_rand())
        self.print_result()
        self.empty()
        return True
