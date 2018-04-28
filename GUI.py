import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tkinter import *
import segment as seg
import RBF
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from sklearn.metrics import confusion_matrix
import math
class InputForm(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Cassification Project'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 500

        self.label1 = QLabel("Choose The classification methon :", self)
        self.classification_method = QComboBox(self)
        self.button = QPushButton('Run', self)
        self.button.setToolTip('Run The Program')




        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.move(10, 260)
        self.classification_method.move(250, 260)
        self.classification_method.addItem("RBF")
        self.classification_method.addItem("MLP")
        self.button.move(200, 400)
        self.button.clicked.connect(self.on_click)

        self.show()

    @pyqtSlot()
    def on_click(self):
        #zaza edit path
        self.Read("/media/abdlrhman/Abdalrhman/Github/Shared_team_work/NeuralNetworksProject/Test_images/")
    def Read(self, test_images_path):

        count = 0
        RBF=False
        MLP=False
        if self.classification_method=="RBF":
            rbf_obj = RBF(22)
            RBF_method=True
        else:
            #zaza MLP object
            MLP = False

        for each in glob(test_images_path + "*"):
            if count % 2 == 0:
                original_img = each #
            else:
                all, pos = seg.segment(original_img, each)
                # create list of ouputs of the classifer
                #zaza
                labels=np.zeros([len(all)])
                im=cv.imread(original_img)
                for j in range(len(pos)): # iterate over objects
                # pass each of all list to the classifier
                 cv.rectangle(im, (pos[j][0],pos[j][2]), (pos[j][1], pos[j][3]), (255,0, 0), 3)
                 font=cv.FONT_HERSHEY_SIMPLEX
                 label="label"
                 cv.putText(im, label, (pos[j][0],pos[j][2]), font, 0.7, (0, 255, 0), 2, cv.LINE_AA)
                cv.imshow("test",im)
                cv.waitKey(0)

            count += 1



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = InputForm()
    sys.exit(app.exec_())