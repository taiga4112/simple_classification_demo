# coding: utf-8

import numpy as np
from scipy.linalg import norm
import cvxopt
import cvxopt.solvers
from pylab import *


def f(x1, w, b):
    return - (w[0] / w[1]) * x1 - (b / w[1])


def kernel(x, y):
    return np.dot(x, y)  # 線形カーネル


def svn_linear_discriminant(plist1, plist2):
    _X = vstack((plist1, plist2))
    _N = len(plist1) + len(plist2)  # データ数

    # ラベルを作成
    t = []
    for i in range(len(plist1)):
        t.append(-1.0)
    for i in range(len(plist2)):
        t.append(1.0)
    t = array(t)

    # ラグランジュ乗数を二次計画法（Quadratic Programming）で求める
    _K = np.zeros((_N, _N))
    for i in range(_N):
        for j in range(_N):
            _K[i, j] = t[i]*t[j]*kernel(_X[i],_X[j])
    _Q = cvxopt.matrix(_K)
    p = cvxopt.matrix(-np.ones(_N))  # -1がN個の列ベクトル
    _G = cvxopt.matrix(np.diag([-1.0] * _N))  # 対角成分が-1のNxN行列
    h = cvxopt.matrix(np.zeros(_N))  # 0がN個の列ベクトル
    _A = cvxopt.matrix(t, (1, _N))  # N個の教師信号が要素の行ベクトル（1xN）
    b = cvxopt.matrix(0.0)  # 定数0.0
    sol = cvxopt.solvers.qp(_Q, p, _G, h, _A, b)  # 二次計画法でラグランジュ乗数aを求める
    a = array(sol['x']).reshape(_N)  # 'x'がaに対応する
    print a

    # サポートベクトルのインデックスを抽出
    _S = []
    for i in range(len(a)):
        if a[i] < 0.0000001: continue
        _S.append(i)

    # wを計算
    w = np.zeros(2)
    for n in _S:
        w += a[n] * t[n] * _X[n]

    # bを計算
    _sum = 0
    for n in _S:
        temp = 0
        for m in _S:
            temp += a[m] * t[m] * kernel(_X[n], _X[m])
        _sum += (t[n] - temp)
    b = _sum / len(_S)
    print _S, b

    return w, b
