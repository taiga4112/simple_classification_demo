# coding: utf-8
import sys
import numpy as np
import cv2
import scd_base
import fisherLinear
import svmLinear

######################
# 画像をtarget_folderに入れて動かすタイプ(テスト用)
######################

# フォルダ内で新しいJPGファイルを自動的に抽出する方法
# target_folder = "./"
# target_file = scd_base.get_newest_file(target_folder)
# print target_file
# if target_file == "":
#     print 'There is no new JPG file in' + target_folder
#     sys.exit()


target_file = "IMG_1059.jpg"

col_img = cv2.imread(target_file, cv2.IMREAD_COLOR)  # 画像から緑と黄の円がある場所の座標を取得する
green_point_list, yellow_point_list = scd_base.get_point_lists(col_img)

print len(green_point_list),len(yellow_point_list)

#########################################################################
# フィッシャーの線形判別
a, b = fisherLinear.fishers_linear_discriminant(green_point_list, yellow_point_list)
px = np.linspace(int(0), int(col_img.shape[1]), 1000)
py = [fisherLinear.f(a,b,x) for x in px]
scd_base.show_point_result(col_img, px, py)
##########################################################################

##########################################################################
# 線形SVM
# w, b = svmLinear.svn_linear_discriminant(green_point_list, yellow_point_list)
# px = np.linspace(int(0), int(col_img.shape[1]), 1000)
# py = [svmLinear.f(x, w, b) for x in px]
# scd_base.show_point_result(col_img, px, py)
##########################################################################

