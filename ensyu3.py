import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import lfilter, remez, ellip, freqz

def calc_y(x: list[int], b: float):
    x1 = 0 #遅延器
    y = np.zeros(10)
    for i in range(10):
        y[i] = x[i] + b * x1
        x1 = y[i]
    return y

def main():
    T = 1 #標本化周期
    x = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # サンプル点列
    xaxis = np.linspace(0, 9, 10)

    """
    (a) b = 0.5  1, 1,5
    """
    plt.subplot(2, 2, 1)  # 2行2列に4分割した画面の左上に表示
    plt.stem(xaxis, x)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("input signal")  # グラフのタイトル
    y = calc_y(x, 0.5)
    plt.subplot(2, 2, 2)  # 2行2列に4分割した画面の左上に表示
    plt.stem(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("output signal b = 0.5")  # グラフのタイトル
    y = calc_y(x, 1.0)
    plt.subplot(2, 2, 3)  # 2行2列に4分割した画面の左上に表示
    plt.stem(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("output signal b = 1.0")  # グラフのタイトル
    y = calc_y(x, 1.5)
    plt.subplot(2, 2, 4)  # 2行2列に4分割した画面の左上に表示
    plt.stem(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("output signal b = 1.5")  # グラフのタイトル
    plt.tight_layout()
    plt.savefig("3-a.png")
    plt.show()

    """
    (a) b = -0.5  -1, -1,5
    """
    plt.subplot(2, 2, 1)  # 2行2列に4分割した画面の左上に表示
    plt.stem(xaxis, x)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("input signal")  # グラフのタイトル
    y = calc_y(x, -0.5)
    plt.subplot(2, 2, 2)  # 2行2列に4分割した画面の左上に表示
    plt.stem(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("output signal b = -0.5")  # グラフのタイトル
    y = calc_y(x, -1.0)
    plt.subplot(2, 2, 3)  # 2行2列に4分割した画面の左上に表示
    plt.stem(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("output signal b = -1.0")  # グラフのタイトル
    y = calc_y(x, -1.5)
    plt.subplot(2, 2, 4)  # 2行2列に4分割した画面の左上に表示
    plt.stem(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("output signal b = -1.5")  # グラフのタイトル
    plt.tight_layout()
    plt.savefig("3-b.png")
    plt.show()


if __name__ == '__main__':
    main()
