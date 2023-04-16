import cv2

def smooth_filter():
    img = cv2.imread(r'C:\Users\ywj\Desktop\graduation design\pictures\p1.png')
    height,width,channels = img.shape
    cv2.namedWindow('mywindow', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('mywindow', img)

    blur = cv2.blur(img, (5, 5))  # 均值模糊，图像尺寸越大，模糊核也尽量大，这样效果会比较明显
    cv2.namedWindow('blur', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('blur', blur)

    dst = cv2.medianBlur(img,5)
    cv2.namedWindow('medianBlur',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('medianBlur',dst)
    cv2.imwrite('medianBlur.png',dst)

    cv2.waitKey(0)

if __name__ == '__main__':
    smooth_filter()