# coding: utf-8
import sys
import cv2
import scd_base
import fisherLinear

import wx
app = wx.App()


#######################
# 画像をtarget_folderに入れて動かすタイプ(テスト用)
#######################
# target_folder = "./"
#
# target_file = scd_base.get_newest_file(target_folder)
# print target_file
# if target_file == "":
#     print 'There is no new JPG file in' + target_folder
#     sys.exit()
# col_img = cv2.imread(target_file, cv2.IMREAD_COLOR)  # 画像から緑と黄の円がある場所の座標を取得する
# green_point_list, yellow_point_list = scd_base.get_point_lists(col_img)
#
# # フィッシャーの線形判別によって求めた境界の傾きと切片を求める
# a, b = fisherLinear.fishers_linear_discriminant(green_point_list, yellow_point_list)
#
# # 結果を元画像に上書きして表示 & 保存
# p1 = (int(0), int(fisherLinear.f(a, b, 0)))
# p2 = (int(col_img.shape[1]), int(fisherLinear.f(a, b, col_img.shape[1])))
# scd_base.show_result(col_img, p1, p2)



################################
# Webカメラから直接やる方法
#################################
tmp_file_name = "input.jpg"

cap = cv2.VideoCapture(1) # 0. 前面カメラ, 1. 背面カメラ(Surfaceの場合)
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame) # 画面に表示する

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('s'):
        cv2.imwrite(tmp_file_name,frame)
        col_img = cv2.imread(tmp_file_name, cv2.IMREAD_COLOR)  # 画像から緑と黄の円がある場所の座標を取得する
        green_point_list, yellow_point_list = scd_base.get_point_lists(col_img)

        if len(green_point_list)==0 or len(yellow_point_list)==0:
            wx.MessageBox(u'Error detected green circle or yellow circle', u'Error', wx.ICON_ERROR)
        else:
            # フィッシャーの線形判別によって求めた境界の傾きと切片を求める
            a, b = fisherLinear.fishers_linear_discriminant(green_point_list, yellow_point_list)

            # 結果を元画像に上書きして表示 & 保存
            p1 = (int(0), int(fisherLinear.f(a, b, 0)))
            p2 = (int(col_img.shape[1]), int(fisherLinear.f(a, b, col_img.shape[1])))
            scd_base.show_result(col_img, p1, p2)

        break #[s]を押したらいずれにせよ終了処理に入るものとする(要検討)

cap.release()
cv2.destroyAllWindows()
