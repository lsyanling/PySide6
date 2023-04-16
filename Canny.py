import os
import cv2
import numpy as np

    #读取图像
pic_path = 'quzao.png'
img=cv2.imread(pic_path)
    #缩放图像的大小
    #img=cv2.resize(src=img,dsize=(512,512))
    #对图像进行边缘检测，低阈值=50 高阈值=200
img_canny=cv2.Canny(img,threshold1=70,threshold2=100)
    #显示检测之后的图像
cv2.imshow('image',img_canny)
    #显示原图像
cv2.imshow('src',img)
cv2.imwrite('canny.png',img_canny)
cv2.waitKey(0)


#销毁所有的窗口
#cv2.destroyAllWindows()
