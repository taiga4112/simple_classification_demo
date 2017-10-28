# coding: utf-8
import sys
import cv2
import scd_base
import fisherLinear

target_folder = "./"

target_file = scd_base.get_newest_file(target_folder)
print target_file
if target_file == "":
    print 'There is no new JPG file in' + target_folder
    sys.exit()

col_img = cv2.imread(target_file, cv2.IMREAD_COLOR)  # 画像から緑と黄の円がある場所の座標を取得する
green_point_list, yellow_point_list = scd_base.get_point_lists(col_img)

# フィッシャーの線形判別によって求めた境界の傾きと切片を求める
a, b = fisherLinear.fishers_linear_discriminant(green_point_list, yellow_point_list)

# 結果を元画像に上書きして表示 & 保存
p1 = (int(0), int(fisherLinear.f(a, b, 0)))
p2 = (int(col_img.shape[1]), int(fisherLinear.f(a, b, col_img.shape[1])))
scd_base.show_result(col_img, p1, p2)
