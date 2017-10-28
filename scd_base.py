# coding: utf-8
import os
import sys
import cv2
import numpy as np
import math

final_output_file_name = "result.jpg"


def get_newest_file(target_folder):
    # jpgファイルリストの作成
    jpg_file_list = []
    for x in os.listdir(target_folder):
        if os.path.isfile(target_folder + x):
            if x[-4:] == ".jpg" or x[-4:] == ".JPG":
                if x != final_output_file_name:
                    jpg_file_list.append(x)

    file_name = ""
    _create_epoc_time = 0
    for jpg_file in jpg_file_list:
        create_epoc_time = os.path.getatime(target_folder + jpg_file)
        if create_epoc_time > _create_epoc_time:
            file_name = jpg_file
            _create_epoc_time = create_epoc_time
    return file_name


def get_point_list(_col_img, target):
    round_th = 0.8  # 円判定閾値
    point_list = []
    _, contours, _ = cv2.findContours(target, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        # 検出輪郭の外接円面積area1を計算
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        area1 = (radius * radius) * math.pi
        # 検出輪郭の実面積area0を計算
        area0 = cv2.contourArea(cnt)
        # 面積比率から輪郭の"円形度"を推定
        if (1 - round_th) < area1 / area0 < (1 + round_th) and area1 > 10000.0:
            cv2.circle(_col_img, (int(cx), int(cy)), int(radius), (0, 0, 0), 2)
            print cx, cy, area1
            point_list.append([cx, cy])
    return point_list


def show_result(col_img, p1, p2):
    cv2.line(col_img, p1, p2, (0, 0, 255), 50)
    cv2.namedWindow('RESULT', cv2.WINDOW_NORMAL)
    cv2.imshow('RESULT', col_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(final_output_file_name, col_img)


def get_point_lists(col_img):
    # 入力画像をHSVチャンネルに分解
    img = cv2.cvtColor(col_img, cv2.COLOR_BGR2HSV)
    h = img[:, :, 0]
    s = img[:, :, 1]
    v = img[:, :, 2]

    shape = (img.shape[0], img.shape[1], 1)  # [px, px, dimension]
    green = np.zeros(shape, dtype=np.uint8)
    yellow = np.zeros(shape, dtype=np.uint8)

    # 色相(Hue)による色成分検出; +彩度(S)と明度(V)の閾値判定(パラメータ調整が必要)
    HF = 0.5  # OpenCVはhは1/2掛けで入力する
    green[((h > 70 * HF) & (h < 150 * HF)) & (s > 16) & (v > 10)] = 255  # 110±40°
    yellow[((h > 30 * HF) & (h < 90 * HF)) & (s > 16) & (v > 10)] = 255  # 60±30°

    # 縮小処理によるノイズ除去(10x10円形カーネル) (適用箇所によってはparameter調整が必要かもしれない)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)

    # 座標群を取得する
    green_point_list = get_point_list(col_img, green)
    yellow_point_list = get_point_list(col_img, yellow)
    return green_point_list, yellow_point_list
