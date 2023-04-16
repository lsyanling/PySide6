import  cv2
import numpy as np
path = r"medianBlur.png"
path1 = r"people3.jpg"
img = cv2.imread(path)
im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kernel = np.array([[0, 2, 0], [-2, 1, -2], [0, 2, 0]])#定义卷积核
imageEnhance = cv2.filter2D(img,-1, kernel)#进行卷积运算
print(imageEnhance.shape)
cv2.imshow('zengqiang',imageEnhance)
cv2.imwrite('zengqiang.png',imageEnhance)
cv2.waitKey(0)