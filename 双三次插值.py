import sys
import numpy as np
import cv2
import scipy.io
import math

def rect_to_polar(img,angle):

    #得到后续要用的img和angle，不用管
    img = np.rot90(img,2)
    img = np.flip(img,1)
    # [修订] angle是一个array，不要使用循环，包括后面的循环大部分都可以改
    angle[angle < 0] *= -1
    # for i in range(256):
    #     if angle[i] < 0:
    #         angle[i] = abs(angle[i])
    
    #得到输入图像(矩形图)的行和列数
    # [修订] img.shape返回3个值，考虑版本差异
    height, width = img.shape[0], img.shape[1]
    origin = height
    
    #计算得到输出图像(扇形图)的行数(height_sec)和列数(width_sec)
    h1 = height
    y = h1 * np.sin(angle[255])
    width_sec = int(np.ceil(2 * y))
    h2 = 1
    x2 = h2 * np.cos(angle[255])
    height_sec = int(np.ceil(h1 - x2))

    #得到扇形图中每一点的半径和角度
    # [修订] L是什么，命名应该有意义
    L = np.zeros((height_sec, width_sec))
    angle2 = np.zeros((height_sec, width_sec))
    for i in range(height_sec):
        x = origin - i - 1
        for j in range(width_sec // 2):     #扇形左半边
            y = np.ceil(width_sec / 2) - j - 1
            L[i, j] = np.sqrt(x**2 + y**2)
            angle2[i, j] = np.arctan(y / x)
        for j in range(width_sec // 2 + 1, width_sec):   #扇形右半边
            y = j - np.ceil(width_sec / 2)+1
            L[i, j] = np.sqrt(x**2 + y**2)
            angle2[i, j] = np.arctan(y / x)

        L[i, int(np.ceil(width_sec / 2)) - 1] = np.sqrt(x**2)    #扇形最中间那列，角度为0
        angle2[i,int(np.ceil(width_sec / 2)) - 1] = 0
    
    #通过扇形图中每一点的半径和角度，得到扇形图中每一点，映射到矩形图上的行坐标和列坐标
    index_row = np.zeros((height_sec, width_sec))       #行坐标
    index_col = np.zeros((height_sec, width_sec))       #列坐标
    for i in range(height_sec):
        cnt = 0                             #统计angle2中每一行里小于angle中间波束角度的数量
        for j in range(width_sec // 2):     #扇形左半边
            if angle2[i, j] <= angle[0]:    #只有角度在扇面内的点才考虑
                index_row[i, j] = origin - L[i, j]-1     #矩形图上的行与扇形图上点的半径对应关系
                for m in range(int(np.ceil(width / 2)) - 1):     #对左半边角度
                    if angle[m + 1] <= angle2[i, j] and angle2[i, j] < angle[m]:    #扇形图上点的角度，在angle[m]和angle[m+1]之间，计算在矩形图上的坐标
                        index_col[i, j] = m + 1 - (angle2[i, j] - angle[m + 1]) / (angle[m] - angle[m + 1])
                if angle2[i, j] < angle[int(np.floor(width / 2)) - 1]:    
                    cnt += 1

        for j in range(width_sec // 2 + 1, width_sec):   #扇形右半边
            if angle2[i, j] <= angle[width-1]:
                index_row[i, j] = origin - L[i, j]-1       
                for m in range(int(np.ceil(width / 2)), width - 1):
                    if angle[m] <= angle2[i, j] and angle2[i, j] < angle[m + 1]:
                        index_col[i, j] = m  + (angle2[i, j] - angle[m]) / (angle[m + 1] - angle[m])
                if angle2[i, j] < angle[int(np.ceil(width / 2))]:
                    cnt += 1

        if width_sec % 2 == 1:     #最中间那列单独计算行坐标
            index_row[i, int(np.ceil(width_sec / 2)) - 1] = origin - L[i, int(np.ceil(width_sec / 2)) - 1] - 1
            cnt += 1

        num = 0                    #计算在两个最小angle中间的角度，对应的列坐标(具体为什么这样做我没看懂)
        for j in range(width_sec):
            if angle2[i, j] < angle[int(np.floor(width / 2)) - 1]:
                num += 1       
                if width % 2 == 0:          # width =256
                    index_col[i, j] = int(np.ceil(width / 2))-1 + num / cnt
                else:
                    index_col[i, j] = int(np.floor(width / 2))-1 + num / (cnt / 2)
    

    # 使用 remap 函数将图像从矩形坐标映射到极坐标
    index_col = index_col.astype(np.float32)
    index_row = index_row.astype(np.float32)
    polar_img = cv2.remap(img, index_col, index_row,cv2.INTER_LINEAR)
    return polar_img


if __name__ == "__main__":
    filePath = sys.path[0]
    img = cv2.imread(filePath+r'\pictures\tmp.png')
    filename = filePath+r'\data\beartable.mat'
    mat = scipy.io.loadmat(filename)
    beartable = mat['beartable']

    angle = beartable.flatten()*math.pi/180

    rect_to_polar(img,angle)

    print("结束")