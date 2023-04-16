import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''
def random_noise(img,noise_num):
    
    #添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
    #:param image: 需要加噪的图片
    #:param noise_num: 添加的噪音点数目，一般是上千级别的
    #:return: img_noise
    
    #
    # 参数image：，noise_num：
    img_noise = img
    # cv2.imshow("src", img)
    rows, cols, chn = img_noise.shape
    # 加噪声
    x_size = np.random.randint(2, 3)# 加噪声的方块大小的长宽
    y_size = np.random.randint(2, 3)
    for i in range(noise_num):
        x = np.random.randint(0, rows)#随机生成指定范围的整数
        y = np.random.randint(0, cols)
        img_noise[x:x+x_size, y:y+y_size, :] = 0

    return img_noise
'''
# img_noise = random_noise("32-2.jpg",30)


def put(path):
    # 读取图片
    img = cv2.imread(path)
    #img = random_noise(img,1000)  # 添加噪声，1000是噪声点数

    #src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图片

    # 二值化，之前噪声是0，经过翻转二值化后变成255（白色）
    #ret, src = cv2.threshold(src, 230, 255, cv2.THRESH_BINARY_INV)  # 翻转二值化
    #ret, src = cv2.threshold(src, 102, 255, cv2.THRESH_BINARY)  # 正常二值化
    # 设置卷积核
    kernel = np.ones((2,2), np.uint8)

    # 图像开运算
    res3 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # 图像闭运算去噪
    res4 = cv2.morphologyEx(res3, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('quzao.png',res4)
    # 图像显示
    plt.subplot(121), plt.imshow(img, plt.cm.gray), plt.title('噪声图片'), plt.axis('off')
    plt.subplot(122), plt.imshow(res4, plt.cm.gray), plt.title('去噪'), plt.axis('off')

    # plt.savefig('2.1new-ing.jpg')
    plt.show()
    
# 图像处理函数，要传入路径
put(r'p3.png')
