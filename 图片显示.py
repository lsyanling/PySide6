from PySide6.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QPushButton, QVBoxLayout,QMainWindow
from PySide6.QtGui import QPixmap
from Ui_界面 import Ui_Form
import glob

class MyWindow(QWidget,Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        #folder_name = QFileDialog.getExistingDirectory(self,"选择文件夹","../")

        #btn = QPushButton('显示')
        #self.lb = QLabel()
        #self.lb.setText('显示窗')

        #self.mainLayout = QVBoxLayout()
        #self.mainLayout.addWidget(self.lb)
        #self.mainLayout.addWidget(btn)
        #self.setLayout(self.mainLayout)

        self.pushButton.clicked.connect(self.showImg)

    def showImg(self):
        #fname, _ = QFileDialog.getOpenFileNames(self, 'Open Images', '../')
        fname = glob.glob("C:/Users/ywj/Desktop/graduation design/pictures/*.png")
        path = fname[0]
        pix = QPixmap(path)
        size = pix.size()
        self.label.setGeometry(40,40,size.width(),size.height())
        self.label.setPixmap(pix)


if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
