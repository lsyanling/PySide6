
import random
import sys
import os
import glob

from PySide6.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QPushButton, QVBoxLayout, QMainWindow
from PySide6.QtGui import QPixmap
from Ui_界面 import Ui_Form

class MainWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupUi(self)
        self.connectUI()

    '''
    装载控件，绑定，初始化
    '''
    def connectUI(self):

        self.pictures_path = r'./pictures/'
        self.extension_name = r'*.png'

        self.pushButton.clicked.connect(self.show_image)

    '''
    功能函数
    '''
    def get_pictures_name(self):
        
        files_path = os.path.join(self.pictures_path, self.extension_name)
        files_name = glob.glob(files_path)

        return files_name
    
    def show_image(self):
        file_name = self.get_pictures_name()

        size = len(file_name)

        if size > 0:
            i = random.randint(0, size - 1)
            path = file_name[i]
            pix = QPixmap(path)
            size = pix.size()
            self.label.setGeometry(40, 40, size.width(), size.height())
            self.label.setPixmap(pix)

        # for i in range (len(file_name)):
        #     path = file_name[i]
        #     pix = QPixmap(path)
        #     size = pix.size()
        #     self.label.setGeometry(40, 40, size.width(), size.height())
        #     self.label.setPixmap(pix)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle(" ")
    window.show()
    app.exit(app.exec())
