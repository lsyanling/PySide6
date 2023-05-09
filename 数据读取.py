import struct
import numpy as np
import scipy.io
import cv2
import math
import pandas as pd
import os
from PIL import Image
import time
from scipy.interpolate import LinearNDInterpolator

def getFileHeader(fid):
    LogHeader = {}
    LogHeader['fileHeader'] = format(struct.unpack('I', fid.read(4))[0],'x')

    if LogHeader['fileHeader'] != '11223344':
        raise ValueError('该文件不是Oculus数据文件')
    
    LogHeader['sizeHeader'] = struct.unpack('I', fid.read(4))[0]
    LogHeader['source'] = fid.read(16).decode('ascii')
    LogHeader['version'] = struct.unpack('H', fid.read(2))[0]
    LogHeader['encryption'] = struct.unpack('H', fid.read(2))[0]
    fid.read(4)  # 结构体内数据对齐
    LogHeader['key'] = struct.unpack('Q', fid.read(8))[0]
    LogHeader['time'] = struct.unpack('d', fid.read(8))[0]
    
    return LogHeader

def getItemHeader(fid):
        LogItem = {}
        LogItem['itemHeader'] = format(struct.unpack('I', fid.read(4))[0], 'x')
        if LogItem['itemHeader'] != 'aabbccdd':
            raise ValueError('读取Item错误')

        LogItem['sizeHeader'] = struct.unpack('I', fid.read(4))[0]
        LogItem['type'] = struct.unpack('H', fid.read(2))[0]
        LogItem['version'] = struct.unpack('H', fid.read(2))[0]
        fid.read(4)  # 结构体内数据对齐
        LogItem['time'] = struct.unpack('d', fid.read(8))[0]
        LogItem['compression'] = struct.unpack('H', fid.read(2))[0]
        fid.read(6)  # 结构体内数据对齐
        LogItem['originalSize'] = struct.unpack('I', fid.read(4))[0]
        LogItem['payloadSize'] = struct.unpack('I', fid.read(4))[0]

        return LogItem

def getOculusMessageHeader(fid):
    OculusMessageHeader = {}

    OculusMessageHeader['oculusId'] = format(struct.unpack('H', fid.read(2))[0], 'x')  #默认转换成小写字母字符串
    if OculusMessageHeader['oculusId'] != '4f53':
        raise ValueError('获取OculusMessageHeader错误')

    OculusMessageHeader['srcDeviceId'] = struct.unpack('H', fid.read(2))[0]
    OculusMessageHeader['dstDeviceId'] = struct.unpack('H', fid.read(2))[0]
    OculusMessageHeader['msgId'] = struct.unpack('H', fid.read(2))[0]
    OculusMessageHeader['msgVersion'] = struct.unpack('H', fid.read(2))[0]
    OculusMessageHeader['payloadSize'] = struct.unpack('I', fid.read(4))[0]
    OculusMessageHeader['spare2'] = struct.unpack('H', fid.read(2))[0]

    return OculusMessageHeader

def getOculusSimplePingResult(fid):
    OculusMessageHeader = getOculusMessageHeader(fid)
    OculusSimpleFireMessage = {}
    OculusSimplePingResult = {}
    
    OculusSimpleFireMessage['head'] = OculusMessageHeader
    OculusSimpleFireMessage['masterMode'] = struct.unpack('B',fid.read(1))[0]
    OculusSimpleFireMessage['pingRate'] = struct.unpack('B',fid.read(1))[0]
    OculusSimpleFireMessage['networkSpeed'] = struct.unpack('B',fid.read(1))[0]
    OculusSimpleFireMessage['gammaCorrection'] = struct.unpack('B',fid.read(1))[0]
    OculusSimpleFireMessage['flags'] = struct.unpack('B',fid.read(1))[0]
    OculusSimpleFireMessage['range'] = struct.unpack('Q',fid.read(8))[0]
    OculusSimpleFireMessage['gainPercent'] = struct.unpack('Q',fid.read(8))[0]
    OculusSimpleFireMessage['speedOfSound'] = struct.unpack('Q',fid.read(8))[0]
    OculusSimpleFireMessage['salinity'] = struct.unpack('Q',fid.read(8))[0]

    OculusSimplePingResult['fireMessage'] = OculusSimpleFireMessage
    OculusSimplePingResult['pingId'] = struct.unpack('I', fid.read(4))[0]
    OculusSimplePingResult['status'] = struct.unpack('I', fid.read(4))[0]
    OculusSimplePingResult['frequency'] = struct.unpack('d', fid.read(8))[0]
    OculusSimplePingResult['temperature'] = struct.unpack('d', fid.read(8))[0]
    OculusSimplePingResult['pressure'] = struct.unpack('d', fid.read(8))[0]
    OculusSimplePingResult['speeedOfSoundUsed'] = struct.unpack('d', fid.read(8))[0]
    OculusSimplePingResult['pingStartTime'] = struct.unpack('I', fid.read(4))[0]
    OculusSimplePingResult['dataSize'] = struct.unpack('B',fid.read(1))[0]
    OculusSimplePingResult['rangeResolution'] = struct.unpack('d', fid.read(8))[0]
    OculusSimplePingResult['nRanges'] = struct.unpack('H', fid.read(2))[0]
    OculusSimplePingResult['nBeams'] = struct.unpack('H', fid.read(2))[0]
    OculusSimplePingResult['imageOffset'] = struct.unpack('I', fid.read(4))[0]
    OculusSimplePingResult['imageSize'] = struct.unpack('I', fid.read(4))[0]
    OculusSimplePingResult['messageSize'] = struct.unpack('I', fid.read(4))[0]

    return OculusSimplePingResult


def getnDataFrame(fid, frameNum):
    backSize = 122

    fid.seek(0, 0)  # 返回文件头
    getFileHeader(fid)

    #跳过不需要的帧，知道达到指定的frameNum帧，当frameNum=0,表示已经到达了需要处理的帧
    while frameNum:          
        getItemHeader(fid)
        OculusMessageHeader = getOculusMessageHeader(fid)
        fid.seek(OculusMessageHeader['payloadSize'], 1)    #跳过当前帧的数据部分
        frameNum -= 1    #已经处理了一帧

    getItemHeader(fid)
    OculusSimplePingResult = getOculusSimplePingResult(fid)
    fid.seek(OculusSimplePingResult['imageOffset'] - backSize, 1)

    range_num = OculusSimplePingResult['nRanges']
    beam_num = OculusSimplePingResult['nBeams']
    data_frame = np.zeros((range_num, beam_num), np.uint8)    

    for i in range(range_num):
        row = struct.unpack(f'{beam_num}B', fid.read(beam_num))    #对每行，读波束数个字节
        data_frame[i] = row

    return data_frame

def getImgae(fid,frameNum):
    #filepath = r'C:\Users\ywj\Desktop\graduation design\Oculus_20210728_110209.oculus'
    #fid = open(filepath,'rb')
    filename = 'beartable.mat'
    mat = scipy.io.loadmat(filename)
    beartable = mat['beartable']

    angle = beartable.flatten()*math.pi/180
    '''
    for i in range(256):
        if angle[i] < 0:
            angle[i] = abs(angle[i])
    '''
    data_frame = np.array(getnDataFrame(fid,frameNum))
    img = data_frame
    #img = np.rot90(data_frame,2)
    #img = np.flip(data_frame,1)
    height,width= img.shape
    origin_row = height
    
    return img,angle

def square2fan(img,angle):
    height,width = img.shape
    rows,cols = height,int(np.ceil(2*height*math.sin(angle[255])))
    imgOut = np.zeros((rows,cols),np.uint8)
    for i in range(width):
        d = np.cos(angle[i]) * np.arange(0, height)     #行
        lats = np.round(d).astype(np.int32)       
        b = np.sin(angle[i]) * np.arange(0, height) + (cols-1 )/ 2    #列
        lons = np.round(b).astype(np.int32)
        imgOut[(lats, lons)] = img[:,i]
    
    imgOut = np.flip(imgOut, 0)
    return imgOut

def inter(filepath,frameNum):
    fid = open(filepath,'rb')
    img,angle = getImgae(fid,frameNum)
    masked_image = line_division(img)
    imgOut = square2fan(masked_image,angle)
    rows,cols = imgOut.shape
    for row in range(rows):
        ps, pe = 0, 0
        for col in range(cols):
            if imgOut[row, col] > 0:
                if ps == 0:
                    ps, pe = col, col
                else:
                    pe = col
        for col in range(ps-1 ,pe):
            if imgOut[row, col] == 0:
                imgOut[row, col] = imgOut[row, col-1]

    return imgOut

def getimgOut(fid,frameNum,output_folder):
        img,angle = getImgae(fid,frameNum)
        masked_image = line_division(img)
        imgOut = square2fan(masked_image,angle)
        imgOut = imgOut = bilinear_interpolation(imgOut)
        return imgOut


    

def bilinear_interpolation(imgOut):
    rows, cols = imgOut.shape

    # 创建一个与imgOut大小相同的空标志数组，值为True的地方表示对应的imgOut像素值为0
    empty_pixels = imgOut == 0

    # 创建一个坐标网格
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # 获取非空像素的坐标和值
    x_known = x[~empty_pixels]
    y_known = y[~empty_pixels]
    z_known = imgOut[~empty_pixels]

    # 对空白像素进行插值
    x_empty = x[empty_pixels]
    y_empty = y[empty_pixels]
    points_known = np.array([x_known, y_known]).T
    points_empty = np.array([x_empty, y_empty]).T
    #start_time = time.time()  # 记录开始时间
    interpolator = LinearNDInterpolator(points_known, z_known)
    z_empty = interpolator(points_empty)
    #z_empty = scipy.interpolate.griddata(points_known, z_known, points_empty, method='linear')
    #elapsed_time = time.time() - start_time  # 计算运行时间
    #print(f"程序运行时间：{elapsed_time}秒")
    # 将插值结果填充到原图像中
    imgOut[empty_pixels] = z_empty

    return imgOut

def sw(w1):
    w = abs(w1)
    if 0 <= w < 1:
        A = 1 - 2 * w**2 + w**3
    elif 1 <= w < 2:
        A = 4 - 8 * w + 5 * w**2 - w**3
    else:
        A = 0
    return A

def rect_to_polar(img,angle):

    img = np.rot90(img,2)
    img = np.flip(img,1)
    for i in range(256):
        if angle[i] < 0:
            angle[i] = abs(angle[i])
    
    height, width = img.shape
    origin = height
    
    h1 = height
    y = h1 * np.sin(angle[255])
    width_sec = int(np.ceil(2 * y))
    h2 = 1
    x2 = h2 * np.cos(angle[255])
    height_sec = int(np.ceil(h1 - x2))

    
    L = np.zeros((height_sec, width_sec))
    angle2 = np.zeros((height_sec, width_sec))
    for i in range(height_sec):
        x = origin - i - 1
        for j in range(width_sec // 2):
            y = np.ceil(width_sec / 2) - j - 1
            L[i, j] = np.sqrt(x**2 + y**2)
            angle2[i, j] = np.arctan(y / x)
        for j in range(width_sec // 2 + 1, width_sec):
            y = j - np.ceil(width_sec / 2)+1
            L[i, j] = np.sqrt(x**2 + y**2)
            angle2[i, j] = np.arctan(y / x)

        L[i, int(np.ceil(width_sec / 2)) - 1] = np.sqrt(x**2)
        angle2[i,int(np.ceil(width_sec / 2)) - 1] = 0
    

    index_row = np.zeros((height_sec, width_sec))
    index_col = np.zeros((height_sec, width_sec))
    for i in range(height_sec):
        cnt = 0
        for j in range(width_sec // 2):
            if angle2[i, j] <= angle[0]:
                index_row[i, j] = origin - L[i, j]-1
                for m in range(int(np.ceil(width / 2)) - 1):
                    if angle[m + 1] <= angle2[i, j] and angle2[i, j] < angle[m]:
                        index_col[i, j] = m + 1 - (angle2[i, j] - angle[m + 1]) / (angle[m] - angle[m + 1])
                if angle2[i, j] < angle[int(np.floor(width / 2)) - 1]:
                    
                    if width % 2 == 0:
                        index_col[i, j] = int(np.floor(width / 2)) - 1 + 1 / 2 - angle2[i, j] / angle[int(np.floor(width / 2)) - 1]
                    else:
                        index_col[i, j] = int(np.ceil(width / 2)) - angle2[i, j] / angle[int(np.floor(width / 2)) - 1]
                    
                    cnt += 1

        for j in range(width_sec // 2 + 1, width_sec):
            if angle2[i, j] <= angle[width-1]:
                index_row[i, j] = origin - L[i, j]-1       ##最后要重点检查
                for m in range(int(np.ceil(width / 2)), width - 1):
                    if angle[m] <= angle2[i, j] and angle2[i, j] < angle[m + 1]:
                        index_col[i, j] = m  + (angle2[i, j] - angle[m]) / (angle[m + 1] - angle[m])
                if angle2[i, j] < angle[int(np.ceil(width / 2))]:
                    
                    if width % 2 == 0:
                        index_col[i, j] = int(np.floor(width / 2)) -1 + 1 / 2 + angle2[i, j] / angle[int(np.ceil(width / 2))]
                    else:
                        index_col[i, j] = int(np.ceil(width / 2)) + angle2[i, j] / angle[int(np.ceil(width / 2))]
                    
                    cnt += 1

        if width_sec % 2 == 1:
            index_row[i, int(np.ceil(width_sec / 2)) - 1] = origin - L[i, int(np.ceil(width_sec / 2)) - 1] - 1
            cnt += 1

        num = 0
        for j in range(width_sec):
            if angle2[i, j] < angle[int(np.floor(width / 2)) - 1]:
                num += 1
                if width % 2 == 0:          # width =256
                    index_col[i, j] = int(np.ceil(width / 2))-1 + num / cnt
                else:
                    index_col[i, j] = int(np.floor(width / 2))-1 + num / (cnt / 2)
    
    '''
    # 使用 remap 函数将图像从矩形坐标映射到极坐标
    index_col = index_col.astype(np.float32)
    index_row = index_row.astype(np.float32)
    polar_img = cv2.remap(img, index_col, index_row,cv2.INTER_LINEAR)
    return polar_img
    '''
    f = img.astype(np.float64)
    f1 = np.pad(f, ((2, 2), (2, 2)), mode='edge')
    g1 = np.zeros((height_sec, width_sec))

    for i in range(height_sec):
        for j in range(width_sec):
            if 0 <= index_row[i, j] < height:
                i1 = int(np.floor(index_row[i, j])) + 1
                j1 = int(np.floor(index_col[i, j])) + 1
                if i1 > 0 and j1 > 0:
                    u = index_row[i, j] - np.floor(index_row[i, j])
                    v = index_col[i, j] - np.floor(index_col[i, j])
                    A = np.array([sw(1 + u), sw(u), sw(1 - u), sw(2 - u)])
                    C = np.array([[sw(1 + v)], [sw(v)], [sw(1 - v)], [sw(2 - v)]])
                    B = np.array([
                        [f1[i1 - 1, j1 - 1], f1[i1 - 1, j1], f1[i1 - 1, j1 + 1], f1[i1 - 1, j1 + 2]],
                        [f1[i1, j1 - 1], f1[i1, j1], f1[i1, j1 + 1], f1[i1, j1 + 2]],
                        [f1[i1 + 1, j1 - 1], f1[i1 + 1, j1], f1[i1 + 1, j1 + 1], f1[i1 + 1, j1 + 2]],
                        [f1[i1 + 2, j1 - 1], f1[i1 + 2, j1], f1[i1 + 2, j1 + 1], f1[i1 + 2, j1 + 2]]
                    ])
                    g1[i, j] = (A @ B @ C).item()

    return g1/255

def line_division(image):
    # 1. 使用Canny算法进行边缘检测
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # 2. 应用Hough变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # 3. 筛选角度近似为90度的直线
    filtered_lines = []
    angle_threshold = 5  # 设定角度阈值（以度为单位）
    for line in lines:
        rho, theta = line[0]
        degrees = np.rad2deg(theta)
        if abs(degrees - 90) <= angle_threshold:
            filtered_lines.append(line)

    # 4. 找到位于最下方的直线
    lowest_line = None
    lowest_y = -np.inf
    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        y0 = b * rho
        if y0 > lowest_y:
            lowest_y = y0
            lowest_line = line

    
    '''
    # 5. 绘制找到的直线并显示
    if lowest_line is not None:
        rho, theta = lowest_line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (255), 2)
    '''

    # 5. 降低直线上方区域的对比度
    contrast_decrease_factor = 0.5  # 设置对比度降低因子（0到1之间）
    modified_image = image.copy()
    if lowest_line is not None:
        rho, theta = lowest_line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # 使用NumPy的广播功能高效地降低对比度
        yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
        mask = yy < y0
        modified_image[mask] = (image[mask] * contrast_decrease_factor).astype(np.uint8)
    
    return modified_image


if __name__ == "__main__":
    filepath = r'C:\Users\ywj\Desktop\graduation design\Oculus_20210728_110209.oculus'
    output_folder = r'C:\Users\ywj\Desktop\imgsOut'
    fid = open(filepath,'rb')
    #getimgOut(filepath,output_folder)
    frameNum = 1190
    img,angle = getImgae(fid,frameNum)
    modified_image = line_division(img)
    #ploar = rect_to_polar(img,angle)
    imgOut = square2fan(modified_image,angle)
    imgOut = bilinear_interpolation(imgOut)
    #imgOut = linear_interpolation(imgOut)
    #imgOut = inter(filepath,frameNum)

    #imgOut = opencvinter(imgOut)
    #g1 = bicubic_rec2sec_lvzheng(img, angle)

  