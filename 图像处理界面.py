import os
import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFileDialog, QSizePolicy
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from natsort import natsorted
from Hough_2 import imgEnhance, detect_line, get_distance_and_degree


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('图片查看器')

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        font = QFont()
        font.setPointSize(14)  # 设置字体大小为14

        self.label_1 = QLabel(self)
        self.label_1.setFixedSize(800, 450)  # 设置label_1的大小
        self.label_1.setStyleSheet("border: 1px solid black;")  # 给label_1添加黑色边框
        left_layout.addWidget(self.label_1)

        self.label_2 = QLabel(self)
        self.label_2.setFixedSize(800, 150)  # 设置label_2的大小
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("border: 1px solid black;")  # 给label_2添加黑色边框

        left_layout.addWidget(self.label_2)

        button_layout = QHBoxLayout()

        self.btn_replay = QPushButton('回放', self)
        self.btn_replay.clicked.connect(self.replay)
        button_layout.addWidget(self.btn_replay)

        self.btn_start = QPushButton('开始', self)
        self.btn_start.clicked.connect(self.start)
        button_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton('暂停', self)
        self.btn_pause.clicked.connect(self.pause)
        button_layout.addWidget(self.btn_pause)

        left_layout.addLayout(button_layout)

        main_layout.addLayout(left_layout)

        button_right_layout = QVBoxLayout()

        self.btn_prev = QPushButton('上一张', self)
        self.btn_prev.clicked.connect(self.prev_image)
        button_right_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton('下一张', self)
        self.btn_next.clicked.connect(self.next_image)
        button_right_layout.addWidget(self.btn_next)

        self.btn_detect_wall = QPushButton('洞壁检测', self)
        self.btn_detect_wall.clicked.connect(self.detect_wall)
        button_right_layout.addWidget(self.btn_detect_wall)

        self.btn_distance_and_direction = QPushButton('距离和方位', self)
        self.btn_distance_and_direction.clicked.connect(self.distance_and_direction)
        button_right_layout.addWidget(self.btn_distance_and_direction)

        main_layout.addLayout(button_right_layout)

        self.setLayout(main_layout)

        self.image_folder = None
        self.image_list = []
        self.current_image_index = -1
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_image)

    def replay(self):
        self.image_folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if self.image_folder:
            self.image_list = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            self.image_list = natsorted(self.image_list)
            self.current_image_index = 0
            self.show_image()

    def start(self):
        if self.image_folder and self.image_list:
            self.timer.start(100)

    def pause(self):
        self.timer.stop()

    def prev_image(self):
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.show_image()

    def detect_wall(self):
        if self.image_list:
            input_image_path = os.path.join(self.image_folder, self.image_list[self.current_image_index])
            img = cv2.imread(input_image_path)
            slopes, intercepts, rotated_line, result = detect_line(img)
        
            # 将 OpenCV 图像转换为 QImage
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            qimage = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)
        
            pixmap = QPixmap.fromImage(qimage)
            self.label_1.setPixmap(pixmap.scaled(self.label_1.size(), Qt.KeepAspectRatio))


    def distance_and_direction(self):
        if self.image_list:
            input_image_path = os.path.join(self.image_folder, self.image_list[self.current_image_index])
            img = cv2.imread(input_image_path)
            slopes, intercepts, rotated_line, _ = detect_line(img)
            m1 = slopes[0]
            lines_distance, distance_1, distance_2, drift_angle = get_distance_and_degree(slopes, intercepts, rotated_line)
        
            # 格式化文本
            distance_and_degree_text = f"两侧洞壁距离为：{lines_distance} 米\n机器人距左侧洞壁距离为：{distance_1} 米\n机器人距右侧洞壁距离为：{distance_2} 米\n"
            if np.isinf(m1):
                distance_and_degree_text += "航向正常"
            elif m1 < 0:
                distance_and_degree_text += f"航向：左偏{drift_angle} 度"
            else:
                distance_and_degree_text += f"航向：右偏{drift_angle} 度"
        
            self.label_2.setText(distance_and_degree_text)


    def show_next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.show_image()
        else:
            self.timer.stop()

    def show_image(self):
        if self.image_list:
            pixmap = QPixmap(os.path.join(self.image_folder, self.image_list[self.current_image_index]))
            self.label_1.setPixmap(pixmap.scaled(self.label_1.size(), Qt.AspectRatioMode.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())
