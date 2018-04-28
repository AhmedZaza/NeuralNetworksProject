import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tkinter import *
import segment as seg
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from sklearn.metrics import confusion_matrix
import math
class InputForm(QWidget):

    def __init__(self):
        super().__init__()
        self.Read("/media/abdlrhman/Abdalrhman/Github/Shared_team_work/NeuralNetworksProject/Test_images/")
        self.title = 'Multi Layer Neural Networks - Task3'
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500
        self.label1 = QLabel("Enter Number Of Hidden Layers :", self)
        self.label2 = QLabel("Enter Number Of Neurons in Each Hidden Layer :", self)
        self.label3 = QLabel("Enter learning Rate :", self)
        self.label4 = QLabel("Enter Number of Epochs :", self)
        self.label5 = QLabel("Choose The Activation Function Type :", self)
        self.label6 = QLabel("Choose The Stopping Criteria :", self)
        self.label7 = QLabel("( In Case Of MSE Selected ) Enter MSE Threshold :", self)
        self.CheckBox = QCheckBox("Bias", self)
        self.textboxHiddenLayers = QLineEdit(self)
        self.textboxNeuronsPerLayer = QLineEdit(self)
        self.textboxLr = QLineEdit(self)
        self.textboxEp = QLineEdit(self)
        self.textboxMSE = QLineEdit(self)
        self.button = QPushButton('Run', self)
        self.button.setToolTip('Run The Program')
        self.ActivationFunctionType = QComboBox(self)
        self.StoppingCriteria = QComboBox(self)
        # input variables
        self.bias = 0
        self.NumberOfLayers = 1
        self.NumberOfNeurons = 1
        self.learning_rate = 0
        self.no_epochs = 1
        self.ActivationFunction = "Sigmoid"
        self.StoppingCondition = "Fix The Number Of Epochs"
        self.MSEThreshold = 0

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.move(10, 20)

        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.move(10, 60)

        self.label3.setAlignment(Qt.AlignCenter)
        self.label3.move(10, 100)

        self.label4.setAlignment(Qt.AlignCenter)
        self.label4.move(10, 140)

        self.label5.setAlignment(Qt.AlignCenter)
        self.label5.move(10, 220)

        self.label6.setAlignment(Qt.AlignCenter)
        self.label6.move(10, 260)

        self.label7.setAlignment(Qt.AlignCenter)
        self.label7.move(30, 300)

        self.CheckBox.move(10, 180)

        self.textboxHiddenLayers.move(220, 20)
        self.textboxHiddenLayers.resize(40, 20)

        self.textboxNeuronsPerLayer.move(310, 60)
        self.textboxNeuronsPerLayer.resize(40, 20)

        self.textboxLr.move(180,100)
        self.textboxLr.resize(40, 20)

        self.textboxEp.move(180, 140)
        self.textboxEp.resize(40, 20)

        self.button.move(200, 400)
        self.button.clicked.connect(self.on_click)

        self.ActivationFunctionType.move(250, 220)
        self.ActivationFunctionType.addItem("Sigmoid")
        self.ActivationFunctionType.addItem("Hyperbolic")

        self.StoppingCriteria.move(250, 260)
        self.StoppingCriteria.addItem("Fix The Number Of Epochs")
        self.StoppingCriteria.addItem("MSE")

        self.textboxMSE.resize(40, 20)
        self.textboxMSE.move(350, 300)
        self.show()

    @pyqtSlot()
    def on_click(self):
        self.NumberOfLayers = self.textboxHiddenLayers.text()
        self.NumberOfNeurons = self.textboxNeuronsPerLayer.text()
        self.learning_rate = self.textboxLr.text()
        self.no_epochs = self.textboxEp.text()
        if self.CheckBox.isChecked():
            self.bias = 1
        else:
            self.bias = 0
        self.ActivationFunction = self.ActivationFunctionType.currentText()
        if self.StoppingCriteria.currentText() == "MSE":
            self.StoppingCondition = "MSE"
            self.MSEThreshold = self.textboxMSE.text()
        else:
            self.StoppingCondition = "Fix The Number Of Epochs"
        MyClass = MultiLayerNN(self.NumberOfLayers, self.NumberOfNeurons, self.bias, self.learning_rate,
                               self.no_epochs, self.ActivationFunction, self.StoppingCondition, self.MSEThreshold)
    def Read(self, test_images_path):
        count = 0
        for each in glob(test_images_path + "*"):
            if count % 2 == 0:
                original_img = each #
            else:
                all, pos = seg.segment(original_img, each)
                print(len(pos))

                im=cv.imread(original_img)
                print(pos)
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