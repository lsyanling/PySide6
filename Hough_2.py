import cv2
import numpy as np

def imgEnhance(img):
    median = cv2.medianBlur(img,3)
    kernel = np.array([[0, 1, 0], [-1, 1, -1], [0, 1, 0]])#定义卷积核
    imageEnhance = cv2.filter2D(median,-1, kernel)#进行卷积运算
    #cv2.imshow('1',imageEnhance)
    return imageEnhance

def noise_remove(img):
    # 使用膨胀操作连接断裂的直线
    imageEnhance = imgEnhance(img)
    ret, thresh = cv2.threshold(imageEnhance, 80, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # 使用腐蚀操作恢复原始直线宽度
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    #cv2.imshow('2',eroded)
    #cv2.waitKey(0)
    return eroded

def find_largest_two_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 计算每个连通域的边界矩形的高度
    contour_heights = []
    for contour in contours:
        _, _, _, h = cv2.boundingRect(contour)
        contour_heights.append(h)
    
    # 根据高度对连通域进行排序并选择最高的两个
    largest_two_contours = [contour for _, contour in sorted(zip(contour_heights, contours), key=lambda x: x[0], reverse=True)[:2]]

    return largest_two_contours

def draw_bounding_rectangles(image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def remove_other_contours(image, contours_to_keep):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, contours_to_keep, -1, 255, thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

def skeleton_extraction(img):          # 骨架抽取算法
 
    canvas = np.zeros(img.shape,dtype=np.uint8)                # 将结果保留在这
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))  # 3*3 方形结构元
 
    while img.any():     # 循环迭代
 
        img_open = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)      # 做开运算
        img_diff = img - img_open                                   # 原图 - 开运算图
        canvas = cv2.bitwise_or(canvas,img_diff)                    # 将结果并在一起
        img = cv2.erode(img,kernel)                                 # 腐蚀
 
    dst = canvas                            #  dst 返回的图片
    return dst.astype(np.uint8)

def find_contours(img):
    eroded = noise_remove(img)

    largest_two_contours = find_largest_two_contours(eroded)

    # 保留长度最长的连通域
    longest_contour = [largest_two_contours[0]]
    result_longest = remove_other_contours(eroded, longest_contour)
    ret_1 = skeleton_extraction(result_longest)

    # 保留长度第二长的连通域
    second_longest_contour = [largest_two_contours[1]]
    result_second_longest = remove_other_contours(eroded, second_longest_contour)
    ret_2 = skeleton_extraction(result_second_longest)

    # 保留两个最长的连通域
    result_both = remove_other_contours(eroded, largest_two_contours)
    ret_3 = skeleton_extraction(result_both)

    #result = draw_bounding_rectangles(image, largest_two_contours)     #画上外接矩形

    #cv2.imshow('3',ret_1)
    #cv2.imshow('4',ret_2)
    #cv2.waitKey(0)

    return ret_1,ret_2

#输入为一组直线和阈值，计算直线斜率，并滤除斜率偏离值大于阈值的直线
def outlier_filter(lines, threshold):      
    slopes = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
    
    mean_slope = np.mean(slopes)
    std_dev_slope = np.std(slopes)
    
    filtered_lines = []
    for line, slope in zip(lines, slopes):
        if abs(slope - mean_slope) <= threshold * std_dev_slope:
            filtered_lines.append(line)
    
    return filtered_lines

#将输入的直线最小二乘拟合成为一条直线
def least_squares_fit(lines):
    x = []
    y = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x.extend([x1, x2])
        y.extend([y1, y2])

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Check if the line is vertical (infinite slope)
    if np.isinf(m):
        c = np.mean(x)

    return m, c



def find_contour_centroid(img):
    #image = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    return cx, cy

def rotate_line_around_point(line, angle_degrees, cx, cy):
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])

    rotated_line = np.empty_like(line)
    for i, point in enumerate(line):
        point_vector = np.array([point[0] - cx, point[1] - cy])
        rotated_point_vector = np.matmul(rotation_matrix, point_vector)
        rotated_line[i] = rotated_point_vector + np.array([cx, cy])

    return rotated_line

def create_sector_mask(img):
    center = (339,373)
    radius = 373
    angle_start = 205
    angle_end = 335
    mask = np.zeros_like(img)
    num_points = 360
    polygon_points = [center]

    for angle in np.linspace(angle_start, angle_end, num_points):
        x = int(center[0] + radius * np.cos(np.deg2rad(angle)))
        y = int(center[1] + radius * np.sin(np.deg2rad(angle)))
        polygon_points.append([x, y])

    polygon_points = np.array([polygon_points], dtype=np.int32)
    cv2.fillPoly(mask, polygon_points, (255, 255, 255))

    return mask

def point_to_line_distance(m1,m2, c1, c2,px, py,px0,py0):
    #lines_distance = abs(c2 - c1) / np.sqrt(m1**2 +1)
    proportion = 7.5 / 373
    lines_distance = proportion*abs(m2 * px0 - py0 + c2) / np.sqrt(m2**2 + 1)
    distance_1 = proportion*abs(m1 * px - py + c1) / np.sqrt(m1**2 + 1)
    distance_2 = proportion*abs(m2 * px - py + c1) / np.sqrt(m2**2 + 1)
    lines_distance = round(lines_distance,3)
    distance_1 = round(distance_1,3)
    distance_2 = round(distance_2,3)

    return distance_1,distance_2,lines_distance

def get_distance_and_degree(slopes,intercepts,rotated_line):
    m1 = slopes[0]
    c1 = intercepts[0]
    m2 = (rotated_line[1, 1] - rotated_line[0, 1]) / (rotated_line[1, 0] - rotated_line[0, 0])
    c2 = rotated_line[0, 1] - m2 * rotated_line[0, 0]
    px,py = (339,373)
    py0 = 150
    px0 = int((py0 - c1)) / m1
    distance_1,distance_2,lines_distance = point_to_line_distance(m1,m2,c1,c2,px,py,px0,py0)
    drift_angle = round(90 - abs(np.rad2deg(np.arctan(m1))),3)
    #print(f"两侧洞壁距离为: {lines_distance}米")
    #print(f"机器人距左侧洞壁的距离为: {distance_1}米")
    #print(f"机器人距左侧洞壁的距离为: {distance_2}米")
    #if np.isinf(m1):
        #print("航向正常")
    #elif m1 < 0:
        #print(f"航向：左偏 {drift_angle} 度")
    #else:
        #print(f"航向：右偏 {drift_angle} 度")
    return lines_distance,distance_1,distance_2,drift_angle

def detect_line(img):
    slopes = []
    intercepts = []
    lines = []

    img_1,img_2 = find_contours(img)
    sector_mask = create_sector_mask(img)

    for image in [img_1, img_2]:
        #image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines_data = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, minLineLength=60, maxLineGap=10)

        if lines_data is None:
            print(f"No lines found in img.")
            continue

        filtered_lines = outlier_filter(lines_data, threshold=2)

        if not filtered_lines:
            print(f"No lines found after outlier filtering in img.")
            continue

        m, c = least_squares_fit(filtered_lines)
        slopes.append(m)
        intercepts.append(c)

        y1 = 0
        x1 = int((y1 - c) / m)
        y2 = image.shape[0]
        x2 = int((y2 - c) / m)

        lines.append(np.array([[x1, y1], [x2, y2]]))

    # Rotate the second line around the centroid of the connected component in the second image
    cx, cy = find_contour_centroid(img)
    #m2 = slopes[0]
    #c2 = cy - m2*cx
    #y1 = 0
    #x1 = int((y1 - c2) / m2)
    #y2 = image.shape[0]
    #x2 = int((y2 - c2) / m2)
    #lines[1] = (np.array([[x1, y1], [x2, y2]]))
    angle_degrees = np.rad2deg(np.arctan(slopes[0]) - np.arctan(slopes[1]))
    rotated_line = rotate_line_around_point(lines[1], angle_degrees, cx, cy)

    for i, line in enumerate([lines[0], rotated_line]):
        cv2.line(img, tuple(line[0]), tuple(line[1]), (0, 0, 255), 2)

    sector_mask = create_sector_mask(img)

    result = cv2.bitwise_and(img, sector_mask)
    #cv2.imshow("Result", result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return slopes,intercepts,rotated_line,result



if __name__ == "__main__":
    img = cv2.imread(r'C:\Users\ywj\Desktop\graduation design\pictures\p500.png')
    #image_path1 = "gujia_1.png"
    #image_path2 = "gujia_2.png"
    #orimage = cv2.imread(r'C:\Users\ywj\Desktop\graduation design\pictures\p1186.png')
    #h,w,c = orimage.shape
    #sector_mask = create_sector_mask(img)
    slopes,intercepts,rotated_line,result= detect_line(img)
    get_distance_and_degree(slopes,intercepts,rotated_line)
