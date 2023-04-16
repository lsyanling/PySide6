import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setWindowTitle(" ")
    win.show()
    app.exit(app.exec_())
