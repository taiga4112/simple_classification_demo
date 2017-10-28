# coding: utf-8
import numpy as np


def f(_a, _b, _x):
    return _a * _x + _b


def fishers_linear_discriminant(plist1, plist2):
    # 各クラスの平均をプロット
    m1 = np.mean(plist1, axis=0)
    m2 = np.mean(plist2, axis=0)

    # 総クラス内共分散行列を計算
    sw = np.zeros((2, 2))
    for n in range(len(plist1)):
        xn = np.matrix(plist1[n]).reshape(2, 1)
        m1 = np.matrix(m1).reshape(2, 1)
        sw += (xn - m1) * np.transpose(xn - m1)
    for n in range(len(plist2)):
        xn = np.matrix(plist2[n]).reshape(2, 1)
        m2 = np.matrix(m2).reshape(2, 1)
        sw += (xn - m2) * np.transpose(xn - m2)
    sw_inv = np.linalg.inv(sw)
    w = sw_inv * (m2 - m1)

    # 識別境界を求める
    a = - (w[0, 0] / w[1, 0])
    m = (m1 + m2) / 2
    b = -a * m[0, 0] + m[1, 0]
    return a, b