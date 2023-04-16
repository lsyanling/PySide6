import cv2
import numpy as np

def find_parallel_lines(image_path, rho_accuracy, theta_accuracy, threshold, min_parallel_angle, max_parallel_angle):
    # 读取图片
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray_image, 100, 200)

    # 霍夫变换
    lines = cv2.HoughLines(edges, rho_accuracy, np.pi / 180 * theta_accuracy, threshold)

    # 查找近似平行线
    parallel_lines = []
    for line1 in lines:
        rho1, theta1 = line1[0]
        for line2 in lines:
            rho2, theta2 = line2[0]
            angle_diff = abs(theta1 - theta2)
            if min_parallel_angle <= angle_diff <= max_parallel_angle:
                parallel_lines.append((line1, line2))

    return parallel_lines

def draw_lines(image_path, parallel_lines):
    image = cv2.imread(image_path)
    for line_pair in parallel_lines:
        for line in line_pair:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Parallel Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "quzao.png"
    rho_accuracy = 1
    theta_accuracy = 1
    threshold = 100
    min_parallel_angle = 175
    max_parallel_angle = 185

    parallel_lines = find_parallel_lines(image_path, rho_accuracy, theta_accuracy, threshold, min_parallel_angle, max_parallel_angle)
    draw_lines(image_path, parallel_lines)



    

