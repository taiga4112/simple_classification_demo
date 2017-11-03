# coding: utf-8
import numpy as np
import cv2
import scd_base
import fisherLinear

import wx
app = wx.App()

################################
# Webカメラから直接画像を取得する方法
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

        if len(green_point_list)==0 and len(yellow_point_list)==0:
            wx.MessageBox(u'No green circle and yellow circle', u'Error', wx.ICON_ERROR)
        elif len(green_point_list)==0:
            wx.MessageBox(u'No green circle', u'Error', wx.ICON_ERROR)
        elif len(yellow_point_list)==0:
            wx.MessageBox(u'No yellow circle', u'Error', wx.ICON_ERROR)
        else:
            # フィッシャーの線形判別によって求めた境界の傾きと切片を求める
            a, b = fisherLinear.fishers_linear_discriminant(green_point_list, yellow_point_list)
            px = np.linspace(int(0), int(col_img.shape[1]), 1000)
            py = [fisherLinear.f(a, b, x) for x in px]
            scd_base.show_point_result(col_img, px, py)

        break #[s]を押したらいずれにせよ終了処理に入るものとする(要検討)

cap.release()
cv2.destroyAllWindows()
