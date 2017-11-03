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
    X = vstack((plist1, plist2))
    N = len(plist1) + len(plist2) # データ数

    # ラベルを作成
    t = []
    for i in range(len(plist1)):
        t.append(-1.0)
    for i in range(len(plist2)):
        t.append(1.0)
    t = array(t)

    # ラグランジュ乗数を二次計画法（Quadratic Programming）で求める
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i,j] = t[i]*t[j]*kernel(X[i],X[j])
    Q = cvxopt.matrix(K)
    p = cvxopt.matrix(-np.ones(N))  # -1がN個の列ベクトル
    G = cvxopt.matrix(np.diag([-1.0] * N))  # 対角成分が-1のNxN行列
    h = cvxopt.matrix(np.zeros(N))  # 0がN個の列ベクトル
    A = cvxopt.matrix(t, (1, N))  # N個の教師信号が要素の行ベクトル（1xN）
    b = cvxopt.matrix(0.0)  # 定数0.0
    sol = cvxopt.solvers.qp(Q, p, G, h, A, b)  # 二次計画法でラグランジュ乗数aを求める
    a = array(sol['x']).reshape(N)  # 'x'がaに対応する
    print a

    # サポートベクトルのインデックスを抽出
    S = []
    for i in range(len(a)):
        if a[i] < 0.0000001: continue
        S.append(i)

    # wを計算
    w = np.zeros(2)
    for n in S:
        w += a[n] * t[n] * X[n]

    # bを計算
    sum = 0
    for n in S:
        temp = 0
        for m in S:
            temp += a[m] * t[m] * kernel(X[n], X[m])
        sum += (t[n] - temp)
    b = sum / len(S)
    print S, b

    return w,b


if __name__ == "__main__":
    print 'a'

