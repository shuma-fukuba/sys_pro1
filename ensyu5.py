from cmath import pi
import numpy as np
import matplotlib.pyplot as plt


def filter(x: list[float], M: int, N: int, T):
    y = np.zeros(N)
    for n in range(N):
        sum = 0
        for m in range(M + 1):
            if n * T - m * T < 0:
                continue
            sum += x[(n * T) - (m * T)]
        y[n] = sum / (M + 1)
    return y


def main():
    T = 1
    N = 50
    xaxis = np.arange(N)
    noise = 0.5 * (np.random.rand(N) - 0.5)
    x = np.sin(4 * np.pi * xaxis / N) + noise

    plt.subplot(2, 2, 1)  # 2行2列に4分割した画面の左上に表示
    plt.plot(xaxis, x)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.title("input signal")  # グラフのタイトル

    y = filter(x, 1, N, T)
    plt.subplot(2, 2, 2)  # 2行2列に4分割した画面の左上に表示
    plt.plot(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.ylabel('Amp')
    plt.xlim([0, 50])
    plt.ylim([-1.5, 1.5])
    plt.title("output signal M = 1")  # グラフのタイトル

    y = filter(x, 5, N, T)
    plt.subplot(2, 2, 3)  # 2行2列に4分割した画面の左上に表示
    plt.plot(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.ylabel('Amp')
    plt.xlim([0, 50])
    plt.ylim([-1.5, 1.5])
    plt.title("output signal M = 5")  # グラフのタイトル

    y = filter(x, 15, N, T)
    plt.subplot(2, 2, 4)  # 2行2列に4分割した画面の左上に表示
    plt.plot(xaxis, y)  # 入力信号のプロット
    plt.xlabel("Time $nT$ [s]")  # X軸のラベル
    plt.ylabel('Amp')
    plt.xlim([0, 50])
    plt.ylim([-1.5, 1.5])
    plt.title("output signal M = 15")  # グラフのタイトル

    plt.tight_layout()
    plt.savefig("5.png")
    plt.show()

if __name__ == '__main__':
    main()
