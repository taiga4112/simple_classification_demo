# coding: utf-8
import sys
import cv2
import numpy as np

import scd_base

target_folder = "./"


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


if __name__ == '__main__':
    target_file = scd_base.get_newest_file(target_folder)
    print target_file
    if target_file == "":
        print 'There is no new JPG file in' + target_folder
        sys.exit()

    col_img = cv2.imread(target_file, cv2.IMREAD_COLOR)
    # 画像から緑と黄の円がある場所の座標を取得する
    green_point_list, yellow_point_list = scd_base.get_point_lists(col_img)

    # フィッシャーの線形判別によって求めた境界の傾きと切片を求める
    a, b = fishers_linear_discriminant(green_point_list, yellow_point_list)

    # 結果を元画像に上書きして表示 & 保存
    p1 = (int(0), int(f(a, b, 0)))
    p2 = (int(col_img.shape[1]), int(f(a, b, col_img.shape[1])))
    scd_base.show_result(col_img, p1, p2)
