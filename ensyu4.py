from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import lfilter, remez, ellip, freqz

# amp
def amp(w, b=0.5, T=1, P=100):
    a = np.zeros(P)
    for i in range(P):
        denominator = np.sqrt(1 + b ** 2 - 2 * b * np.cos(w[i] * T))
        a[i] = 1 / denominator
    return a


# theta
def theta(w, b=0.5, T=1, P=100):
    results =  np.zeros(P)
    for i in range(P):
        molecule = b * np.sin(w[i] * T)
        denominator = b * np.cos(w[i] * T) - 1
        results[i] = np.arctan(molecule / denominator)
        results[i] = (results[i] / (2 * pi)) * 360
    return results



def main():
    f = np.linspace(0, 0.5, 100)  # 0Hzから0.5Hzまで100点とる
    w = 2 * np.pi * f  # 角周波数に変換

    # (a)
    bs = np.array([0.5, 0.9, -0.5, -0.9])
    for i, b in enumerate(bs):
        a = amp(w=w, b=b)
        plt.subplot(2, 2, i+1)  # 2行2列に4分割した画面の左上に表示
        plt.plot(f, a)  # 軸fでプロット
        plt.xlim([0, 0.5])  # x軸の表示範囲
        plt.xlabel("Frequency $f$ [Hz]")  # x軸のラベル
        plt.ylabel('amp(w)')
        plt.title(f"$b = {b}$")  # グラフのタイトル
    plt.tight_layout()
    plt.savefig("4-a.png")
    plt.show()

    # (b)
    Ts = np.array([0.5, 1, 2, 4])   
    for i, T in enumerate(Ts):
        a = amp(w=w, T=T)
        plt.subplot(2, 2, i+1)  # 2行2列に4分割した画面の左上に表示
        plt.plot(f, a)  # 軸fでプロット
        plt.xlim([0, 0.5])  # x軸の表示範囲
        plt.xlabel("Frequency $f$ [Hz]")  # x軸のラベル
        plt.ylabel('amp(w)')
        plt.title(f"$T = {T}$")  # グラフのタイトル
    plt.tight_layout()
    plt.savefig("4-b.png")
    plt.show()

    # (c)
    f = f = np.linspace(-1.0, 1.0, 400)
    w = 2 * np.pi * f  # 角周波数に変換

    b = 0.5
    a = amp(w=w, b=b, P=400)
    plt.subplot(2, 2, 1)
    plt.plot(f, a)
    plt.xlim([-1.0, 1.0])
    plt.xlabel("Frequency $f$ [Hz]")  # x軸のラベル
    plt.ylabel('amp(w)')
    plt.title(f"$b = {b}$")  # グラフのタイトル

    a = theta(w=w, b=b, P=400)
    plt.subplot(2, 2, 2)
    plt.plot(f, a)
    plt.xlim([-1.0, 1.0])
    plt.xlabel("Frequency $f$ [Hz]")  # x軸のラベル
    plt.ylabel('theta(degree)')
    plt.title(f"$b = {b}$")  # グラフのタイトル

    b = 0.9
    a = amp(w=w, b=b, P=400)
    plt.subplot(2, 2, 3)
    plt.plot(f, a)
    plt.xlim([-1.0, 1.0])
    plt.xlabel("Frequency $f$ [Hz]")  # x軸のラベル
    plt.ylabel('amp(w)')
    plt.title(f"$b = {b}$")  # グラフのタイトル

    a = theta(w=w, b=b, P=400)
    plt.subplot(2, 2, 4)
    plt.plot(f, a)
    plt.xlim([-1.0, 1.0])
    plt.xlabel("Frequency $f$ [Hz]")  # x軸のラベル]
    plt.ylabel('theta(degree)')
    plt.title(f"$b = {b}$")  # グラフのタイトル


    plt.tight_layout()
    plt.savefig("4-c.png")
    plt.show()


if __name__ == '__main__':
    main()
