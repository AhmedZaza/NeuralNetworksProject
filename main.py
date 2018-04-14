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
# cat = 1
# laptop = 2
# apple = 3
# car = 4
# helicopter = 5
class ObjectRecognizer:
    def __init__(self):
        self.TestingData = []
        self.TrainingData = []
        self.TrainingLabels = []
        self.TestingLabels = []

    def Normalize(self):
        # Standardizing the features
        self.TrainingData = StandardScaler().fit_transform(self.TrainingData)
        self.TestingData = StandardScaler().fit_transform(self.TestingData)
        #for i in range(0, 2500):
        #    mean = np.mean(self.TrainingData[:, i])
        #    max = np.max(self.TrainingData[:, i])
        #    self.TrainingData[:, i] -= mean
        #    self.TrainingData[:, i] /= max

    def Read(self, train_path, test_path):
        # Reading Training Data
        for each in glob(train_path + "*"):
            take = False
            st = ""
            for ch in each:
                if ch == ' ' and st != "":
                    break
                if ch == '.':
                    break
                if take and ch != ' ':
                    st += ch
                if ch == '-':
                    take = True
            if st == "Cat":
                self.TrainingLabels.append(1)
            elif st == "Laptop":
                self.TrainingLabels.append(2)
            elif st == "Apple":
                self.TrainingLabels.append(3)
            elif st == "Car":
                self.TrainingLabels.append(4)
            elif st == "Helicopter":
                self.TrainingLabels.append(5)
            im = cv.imread(each, 0)
            im = cv.resize(im, (50, 50))
            final_data = np.reshape(im, 2500)
            self.TrainingData.append(final_data)
        self.TrainingData = np.array(self.TrainingData, dtype='float64')
        # Reading Testing Data
        count = 0
        for each in glob(test_path + "*"):
            if count % 2 == 0:
                original_path = each
            else:
                all, pos = seg.segment(original_path, each)
                stri = ""
                for i in each.split('/')[-1]:
                    if i == ' ':
                        break
                    stri += i
                # cat = 1
                # laptop = 2
                # apple = 3
                # car = 4
                # helicopter = 5
                if stri == "T1":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(1)
                    self.TestingLabels.append(2)
                elif stri == "T2":
                    self.TestingData.append(all[1])
                    self.TestingData.append(all[2])
                    self.TestingLabels.append(1)
                    self.TestingLabels.append(2)
                elif stri == "T3":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(2)
                    self.TestingLabels.append(1)
                elif stri == "T4":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingData.append(all[2])
                    self.TestingLabels.append(4)
                    self.TestingLabels.append(4)
                    self.TestingLabels.append(1)
                elif stri == "T5":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(2)
                    self.TestingLabels.append(3)
                elif stri == "T6":
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(5)
                elif stri == "T7":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingData.append(all[2])
                    self.TestingLabels.append(5)
                    self.TestingLabels.append(1)
                    self.TestingLabels.append(3)
                elif stri == "T8":
                    self.TestingData.append(all[0])
                    self.TestingLabels.append(4)
                elif stri == "T9":
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(3)
                elif stri == "T10":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(4)
                    self.TestingLabels.append(1)
                elif stri == "T11":
                    self.TestingData.append(all[0])
                    self.TestingLabels.append(1)
                elif stri == "T12":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(2)
                    self.TestingLabels.append(1)
                elif stri == "T13":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(5)
                    self.TestingLabels.append(4)
                elif stri == "T14":
                    self.TestingData.append(all[0])
                    self.TestingData.append(all[1])
                    self.TestingLabels.append(4)
                    self.TestingLabels.append(4)
            count += 1
        self.TestingData = np.array(self.TestingData, dtype='float64')

    def calculate_pca(self):
        pca = PCA(25)
        pca.fit(self.TrainingData)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
        self.TrainingData = pca.transform(self.TrainingData)
        self.TestingData = pca.transform(self.TestingData)

    def Run(self):
        self.Read("/Users/mac/PycharmProjects/NNProject/Training/", "/Users/mac/PycharmProjects/NNProject/Testing/")
        self.Normalize()
        self.calculate_pca()
        cv.waitKey(0)

class MultiLayerNN:
    def __init__(self, no_layers, no_neu, b, lr, no_ep, af, sc, mse):
        # System Variables
        self.Features = ObjectRecognizer()
        self.Features.Run()
        self.TrainingData = self.Features.TrainingData
        self.TestingData = self.Features.TestingData
        self.TestingLabels = self.Features.TestingLabels
        self.TrainingLabels = self.Features.TrainingLabels
        self.bias = int(b)
        self.NumberOfLayers = int(no_layers)
        x = no_neu.split(',')
        z = []
        for i in x:
            z.append(int(i))
        arr = np.array(z)
        if arr.shape[0] == 1:
            self.NumberOfNeurons = np.full(self.NumberOfLayers, arr[0])
            self.MaxNeuron = int(arr[0])
        elif arr.shape[0] == self.NumberOfLayers:
            self.NumberOfNeurons = np.zeros(self.NumberOfLayers, int)
            self.MaxNeuron = 0
            for i in range(0, self.NumberOfLayers):
                self.NumberOfNeurons[i] = arr[i]
                self.MaxNeuron = max(self.MaxNeuron, arr[i])
        else:
            return
        self.learning_rate = float(lr)
        self.ActivationFunction = af
        self.StoppingCondition = sc
        if sc != "MSE":
            self.no_epochs = int(no_ep)
        self.MSEThreshold = float(mse)
        self.NumberOfFeatures = 25
        self.NumberOfClasses = 5
        self.Weights = np.zeros((self.NumberOfLayers + 1, max(self.MaxNeuron, self.NumberOfFeatures), max(self.MaxNeuron, self.NumberOfFeatures)))
        self.biasList = np.zeros((self.NumberOfLayers + 1, 1))
        self.Out = np.zeros((self.NumberOfLayers + 1, max(self.MaxNeuron, self.NumberOfFeatures)))
        self.Error = np.zeros((self.NumberOfLayers + 1, max(self.MaxNeuron, self.NumberOfFeatures)))
        self.OutError1 = np.zeros((25, 1))
        self.OutError2 = np.zeros((25, 1))
        self.OutError3 = np.zeros((25, 1))
        self.OutError4 = np.zeros((25, 1))
        self.OutError5 = np.zeros((25, 1))
        self.Epochs = []
        self.initialize()
        self.train()
        self.Run()

    def initialize(self):
        np.random.seed(0)
        for i in range(0, self.Weights.shape[0]):
            for j in range(0, self.Weights.shape[1]):
                self.Weights[i, j] = np.random.uniform(-1, 1)
        if self.bias == 1:
            for i in range(0, self.biasList.shape[0]):
                self.biasList[i] = np.random.uniform(-1, 1)

    def ActFunction(self, vnet):
        if self.ActivationFunction == "Sigmoid":
            return 1 / (1 + math.exp(vnet * -1))
        else:
            return (1 - math.exp(vnet * -1)) / (1 + math.exp(vnet * -1))

    def train(self):
        Epoch_Number = 0
        OK = True
        while OK:
            for index in range(0, self.TrainingData.shape[0]):
                X = self.TrainingData[index]
                D = self.TrainingLabels[index]
                # Forward Step....
                for Level in range(0, self.NumberOfLayers + 1):
                    if Level == 0:
                        Weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                            Vnet = np.sum(Weight[NeuronIndx, 0:25] * X) + self.biasList[Level] * self.bias
                            self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

                    elif Level == self.NumberOfLayers:
                        Weight = self.Weights[Level]
                        X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level - 1]]

                        Vnet = np.sum(Weight[0, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out1 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[1, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out2 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[2, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out3 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[3, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out4 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[4, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out5 = self.ActFunction(Vnet)

                        if D == 1:
                            D1 = 1
                            D2 = 0
                            D3 = 0
                            D4 = 0
                            D5 = 0
                        elif D == 2:
                            D1 = 0
                            D2 = 1
                            D3 = 0
                            D4 = 0
                            D5 = 0
                        elif D == 3:
                            D1 = 0
                            D2 = 0
                            D3 = 1
                            D4 = 0
                            D5 = 0
                        elif D == 4:
                            D1 = 0
                            D2 = 0
                            D3 = 0
                            D4 = 1
                            D5 = 0
                        elif D == 5:
                            D1 = 0
                            D2 = 0
                            D3 = 0
                            D4 = 0
                            D5 = 1

                        self.OutError1[index] = (D1 - Out1)
                        self.OutError2[index] = (D2 - Out2)
                        self.OutError3[index] = (D3 - Out3)
                        self.OutError4[index] = (D4 - Out4)
                        self.OutError5[index] = (D5 - Out5)
                        E1 = (D1 - Out1) * Out1 * (1 - Out1)
                        E2 = (D2 - Out2) * Out2 * (1 - Out2)
                        E3 = (D3 - Out3) * Out3 * (1 - Out3)
                        E4 = (D4 - Out4) * Out4 * (1 - Out4)
                        E5 = (D5 - Out5) * Out5 * (1 - Out5)
                        self.Error[Level, 0] = E1
                        self.Error[Level, 1] = E2
                        self.Error[Level, 2] = E3
                        self.Error[Level, 3] = E4
                        self.Error[Level, 4] = E5
                    else:
                        Weight = self.Weights[Level]
                        X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level - 1]]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                            Vnet = np.sum(Weight[NeuronIndx, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                                Level] * self.bias
                            self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)
                # Backward Step .......
                for Level in range(self.NumberOfLayers, -1, -1):
                    if Level == self.NumberOfLayers:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level - 1]):
                            f = self.Out[Level - 1, NeuronIndx]
                            self.Error[Level - 1, NeuronIndx] = f * (1 - f) * np.sum(
                                self.Error[Level, 0:5] * weight[0:5, NeuronIndx])
                            Temp = weight[0:5, NeuronIndx] + self.learning_rate * self.Error[Level, 0:5] * f
                            self.Weights[Level, 0:5, NeuronIndx] = Temp[0:5]
                        self.biasList[Level] = self.biasList[Level] + np.sum(
                            self.learning_rate * self.Error[Level, 0:5] * self.bias)
                    elif Level == 0:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, 25):
                            X = self.TrainingData[index]
                            NTemp = weight[0:self.NumberOfNeurons[Level], NeuronIndx] + self.learning_rate * self.Error[
                                                                                                             Level, 0:
                                                                                                                    self.NumberOfNeurons[
                                                                                                                        Level]] * \
                                    X[NeuronIndx]
                            self.Weights[Level, 0:self.NumberOfNeurons[Level], NeuronIndx] = NTemp[
                                                                                             0:self.NumberOfNeurons[
                                                                                                 Level]]
                        self.biasList[Level] = self.biasList[Level] + np.sum(
                            self.learning_rate * self.Error[Level, 0:self.NumberOfNeurons[Level]] * self.bias)

                    else:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level - 1]):
                            f = self.Out[Level - 1, NeuronIndx]
                            self.Error[Level - 1, NeuronIndx] = f * (1 - f) * np.sum(
                                self.Error[Level, 0:self.NumberOfNeurons[Level]] * weight[0:self.NumberOfNeurons[Level],
                                                                                   NeuronIndx])
                            Temp = weight[0:self.NumberOfNeurons[Level], NeuronIndx] + self.learning_rate * self.Error[
                                                                                                            Level, 0:
                                                                                                                   self.NumberOfNeurons[
                                                                                                                       Level]] * f
                            self.Weights[Level, 0:self.NumberOfNeurons[Level], NeuronIndx] = Temp[
                                                                                             0:self.NumberOfNeurons[
                                                                                                 Level]]
                        self.biasList[Level] = self.biasList[Level] + np.sum(
                            self.learning_rate * self.Error[Level, 0:self.NumberOfNeurons[Level]] * self.bias)

            MSE1 = 0.5 * np.mean((self.OutError1 ** 2))
            MSE2 = 0.5 * np.mean((self.OutError2 ** 2))
            MSE3 = 0.5 * np.mean((self.OutError3 ** 2))
            MSE4 = 0.5 * np.mean((self.OutError4 ** 2))
            MSE5 = 0.5 * np.mean((self.OutError5 ** 2))

            TotalMSE = MSE1 + MSE2 + MSE3 + MSE4 + MSE5
            print(TotalMSE)
            self.Epochs.append(TotalMSE)
            if self.StoppingCondition == "MSE":
                if TotalMSE <= self.MSEThreshold:
                    break
            else:
                if Epoch_Number == self.no_epochs - 1:
                    break

            Epoch_Number += 1

    def confusion(self, predicted, real):
        con = confusion_matrix(real, predicted)
        print(con)
        acc = 0
        for i in range(0, 5):
            acc += con[i, i]
        return (acc / len(real)) * 100

    def test(self, X):
        # Forward Step....
        for Level in range(0, self.NumberOfLayers + 1):
            if Level == 0:
                Weight = self.Weights[Level]
                for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                    Vnet = np.sum(Weight[NeuronIndx, 0:25] * X) + self.biasList[Level] * self.bias
                    self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

            elif Level == self.NumberOfLayers:
                Weight = self.Weights[Level]
                X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level - 1]]

                Vnet = np.sum(Weight[0, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out1 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[1, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out2 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[2, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out3 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[3, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out4 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[4, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out5 = self.ActFunction(Vnet)

                if Out1 > Out2 and Out1 > Out3 and Out1 > Out4 and Out1 > Out5:
                    return 1
                elif Out2 > Out1 and Out2 > Out3 and Out2 > Out4 and Out2 > Out5:
                    return 2
                elif Out3 > Out1 and Out3 > Out2 and Out3 > Out4 and Out3 > Out5:
                    return 3
                elif Out4 > Out1 and Out4 > Out3 and Out4 > Out2 and Out4 > Out5:
                    return 4
                else:
                    return 5
            else:
                Weight = self.Weights[Level]
                X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level - 1]]
                for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                    Vnet = np.sum(Weight[NeuronIndx, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                        Level] * self.bias
                    self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

    def Run(self):
        predicted = []
        for i in range(0, self.TestingData.shape[0]):
            predicted.append(self.test(self.TestingData[i]))
        pre = np.array(predicted)
        print("Accuracy : ",self.confusion(pre, self.TestingLabels))
        print("# Hidden Layers : ",self.NumberOfLayers)
        print("# Neurons : ",self.NumberOfNeurons[:])
        ep = np.array(self.Epochs)
        epoch = np.zeros((ep.shape[0], 1))
        for i in range(0, ep.shape[0]):
            epoch[i] = i + 1
        plt.plot(epoch, ep)
        plt.xlabel("epoch number ")
        plt.ylabel("MSE")
        plt.title("Learning Curve")
        plt.show()

class InputForm(QWidget):

    def __init__(self):
        super().__init__()
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = InputForm()
    sys.exit(app.exec_())
