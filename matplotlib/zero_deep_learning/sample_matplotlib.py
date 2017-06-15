# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def show_sin_graph():
    '''
    sinのグラフを表示する
    '''
    # 0 -> 6まで0.1刻み
    x = np.arange(0, 6, 0.1)
    y = np.sin(x)

    plt.plot(x, y)
    plt.show()

def show_2_graphs():
    '''
    2つのグラフを表示する
    '''
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.plot(x, y1, label='sin')
    plt.plot(x, y2, linestyle = '--', label='cos')
    # 軸のラベル
    plt.xlabel('x')
    plt.ylabel('y')
    # タイトル
    plt.title('sin & cos')
    # 凡例
    plt.legend()
    plt.show()

def show_image():
    '''
    画像を表示する
    '''
    from matplotlib.image import imread
    img = imread('./viva_beers_logo_purple.png')
    plt.imshow(img)

    plt.show()

if __name__ == '__main__':
    show_sin_graph()
    show_2_graphs()
    show_image()
