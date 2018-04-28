import math
import numpy as np
import  copy
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

class RBF :
        def __init__(self,k,epochs=400):

            self.epochs=epochs
            self.k=k
            #self.load_data()
            self.TrainData= np.genfromtxt("TrainData.txt", delimiter=",")
            self.Train_labels=np.genfromtxt("Train_labels.txt",delimiter=",")

            self.TestData=np.genfromtxt("TestData.txt",delimiter=',')
            self.Test_labels=np.genfromtxt("Test_labels.txt",delimiter=',')

            self.Weight_out=np.zeros([5,k])                      # 2nd level of weights //

            self.hidden_neurons=self.kmean(k,self.TrainData) # initiate hidden neurons k* num of samples
            kmeans = KMeans(n_clusters=17, random_state=0).fit(self.TrainData)
            #self.hidden_neurons=kmeans.cluster_centers_
            #print(kmeans.cluster_centers_)


            #= k*25

            self.hidden_Gaussian=np.zeros(k)
            self.init_sigma()
            self.init_weights()
            self.OutError1 = np.zeros((25, 1))
            self.OutError2 = np.zeros((25, 1))
            self.OutError3 = np.zeros((25, 1))
            self.OutError4 = np.zeros((25, 1))
            self.OutError5 = np.zeros((25, 1))

            #[0, 0: self.NumberOfNeurons[Level - 1]]

        def confusion(self,pred, real):
         con = confusion_matrix(real, pred)
         print(con)
         acc = 0
         for i in range(5):
            acc += con[i, i]
         return (acc / len(real)) * 100

        def calc_Gaussian(self,X,iteration): # guess WRONG ??????????????????
            tmp_hidden_Gaussian=np.zeros([self.k])
            for i in range(self.k): # k
                tmp_hidden_Gaussian[i]=math.exp(-(self.Euclidean_dis(X[iteration],self.hidden_neurons[i])**2)/(2* self.sigma**2))

            return tmp_hidden_Gaussian

        def init_weights(self):

            # out level
            np.random.seed(0)
            for i in range(5):
                for j in range(self.k):
                 self.Weight_out[i][j]=np.random.uniform(-1,1)

        def Euclidean_dis(self,first,second):
            dis=0
            for i in range (len(first)): # number of features
                dis+=(first[i]-second[i])*(first[i]-second[i])
            dis=math.sqrt(dis)
            return  dis

        def init_sigma(self): # ?
            Max_dis=1
            #Max_dis => max distance between any 2 centroids
            max=-100000000000
            for i in range (len(self.hidden_neurons)):
                for j in range(i+1,len(self.hidden_neurons)):
                    cur_dist=self.Euclidean_dis(self.hidden_neurons[i],self.hidden_neurons[j])
                    if(cur_dist>max):
                        max=cur_dist
            Max_dis=max
            self.sigma= Max_dis / math.sqrt(2 * self.k)

        def kmean(self,k,data_set):
            # comment this and add h ,w dimensions

            w=len(data_set[0])  # number of features
            h=len(data_set)     # number of data
            #print(len(data_set))

            self.Matrix=data_set
            #print("===============")
            centroids=[]
            #intalize empty adjancy lists
            for i in range(k):
                tmp_list=[]
                centroids.append(tmp_list)

            for i in range(k):
                for j in range (w):
                  centroids[i].append(self.Matrix[i][j])

            #print(centroids)

            New_centroids = copy.deepcopy(centroids)
            cc=0
            while(1):

                classes = []
                classes.clear()
                # intalize empty adjancy lists
                for i in range(k):
                    tmp_list = []
                    classes.append(tmp_list)

                for i in range(h):  # iterate over the data
                    min = 1000000
                    assigned_cluster = 0  # assume it's always belong to thr 1st cluster
                    for j in range(k):  # iterate over k classes
                        Euclidean = 0
                        for l in range(w):
                            Euclidean += (self.Matrix[i][l] - centroids[j][l]) * (
                                        self.Matrix[i][l] - centroids[j][l])

                        Euclidean = math.sqrt(Euclidean);
                        if (Euclidean < min):
                            min = Euclidean
                            assigned_cluster = j
                    # print(assigned_cluster)
                    classes[assigned_cluster].append(i)

                # print(classes,end='')
                # calculating  new centroid of each class
                #print(centroids)

                for i in range(k):
                    for o in range(w):
                        avg = 0
                        for j in range(len(classes[i])):
                            avg += self.Matrix[classes[i][j]][o]
                        New_centroids[i][o] = avg / len(classes[i])

                #print(centroids)
                #print(New_centroids)
                # check for the stopping condition
                cnt = 0
                cnt2 = 0
                for i in range(k):
                    cnt = 0
                    for j in range(w):
                        if (New_centroids[i][j] - centroids[i][j] <= 0.0):
                            cnt += 1;
                    if (cnt == w):
                        cnt2 += 1;
                if cnt2 == k:
                   #print(New_centroids)
                    return  New_centroids
                    break
                cc+=1
                #print(cc)
                centroids = copy.deepcopy(New_centroids)

        def train(self,learn_rate=0.03,mse_thresh=0.01):

         epoches=self.epochs
         epoch_list = np.zeros([epoches, 1])

         for e in range(epoches):
             #print(self.Weight_out)
             for i in range(len(self.TrainData)): # iterate over samples
                 # net = w * φ T
                 X=self.TrainData
                 cur_hidden_Gaussian = self.calc_Gaussian(X,i)
                 #print(cur_hidden_Gaussian.shape)
                 #print(self.Weight_out[0, 0:self.k].shape)

                 vnet1=np.sum(self.Weight_out[0, 0:self.k] * cur_hidden_Gaussian)
                 vnet2=np.sum(self.Weight_out[1, 0:self.k] * cur_hidden_Gaussian)
                 vnet3=np.sum(self.Weight_out[2, 0:self.k] * cur_hidden_Gaussian)
                 vnet4=np.sum(self.Weight_out[3, 0:self.k] * cur_hidden_Gaussian)
                 vnet5=np.sum(self.Weight_out[4, 0:self.k] * cur_hidden_Gaussian)

                 D=self.Train_labels[i]

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
                 # calc errors
                 self.OutError1[i] = (D1 - vnet1)
                 self.OutError2[i] = (D2 - vnet2)
                 self.OutError3[i] = (D3 - vnet3)
                 self.OutError4[i] = (D4 - vnet4)
                 self.OutError5[i] = (D5 - vnet5)
                 # update weights
                 self.Weight_out[0, 0:self.k] = self.Weight_out[0, 0:self.k] + self.OutError1[i] * learn_rate * cur_hidden_Gaussian
                 self.Weight_out[1, 0:self.k] = self.Weight_out[1, 0:self.k] + self.OutError2[i] * learn_rate * cur_hidden_Gaussian
                 self.Weight_out[2, 0:self.k] = self.Weight_out[2, 0:self.k] + self.OutError3[i] * learn_rate * cur_hidden_Gaussian
                 self.Weight_out[3, 0:self.k] = self.Weight_out[3, 0:self.k] + self.OutError4[i] * learn_rate * cur_hidden_Gaussian
                 self.Weight_out[4, 0:self.k] = self.Weight_out[4, 0:self.k] + self.OutError5[i] * learn_rate * cur_hidden_Gaussian
                 #print (self.Weight_out)
             MSE1 = 0.5 * np.mean((self.OutError1 ** 2))
             MSE2 = 0.5 * np.mean((self.OutError2 ** 2))
             MSE3 = 0.5 * np.mean((self.OutError3 ** 2))
             MSE4 = 0.5 * np.mean((self.OutError4 ** 2))
             MSE5 = 0.5 * np.mean((self.OutError5 ** 2))
             TotalMSE = (MSE1 + MSE2 + MSE3 + MSE4 + MSE5) / 5
             print("epoch",str(e)+"---" , TotalMSE)
             epoch_list[e] = e
             if TotalMSE <= mse_thresh:
                 break

         return epoch_list

        def test(self):
            y = np.zeros([len(self.TestData), 1])

            for i in range(0,len(self.TestData)):
                X=self.TestData
                cur_hidden_Gaussian = self.calc_Gaussian(X,i)
                vnet1 = np.sum(self.Weight_out[0, 0:self.k] * cur_hidden_Gaussian)
                vnet2 = np.sum(self.Weight_out[1, 0:self.k] * cur_hidden_Gaussian)
                vnet3 = np.sum(self.Weight_out[2, 0:self.k] * cur_hidden_Gaussian)
                vnet4 = np.sum(self.Weight_out[3, 0:self.k] * cur_hidden_Gaussian)
                vnet5 = np.sum(self.Weight_out[4, 0:self.k] * cur_hidden_Gaussian)
                if vnet1 > vnet2 and vnet1 > vnet3 and vnet1 > vnet4 and vnet1 > vnet5:
                    y[i]= 1
                elif vnet2 > vnet1 and vnet2 > vnet3 and vnet2 > vnet4 and vnet2 > vnet5:
                    y[i]= 2
                elif vnet3 > vnet1 and vnet3 > vnet2 and vnet3 > vnet4 and vnet3 > vnet5:
                    y[i]= 3
                elif vnet4 > vnet1 and vnet4 > vnet3 and vnet4 > vnet2 and vnet4 > vnet5:
                    y[i]= 4
                else:
                    y[i]= 5
            d=self.Test_labels
            #print(y)
            print("test accuarcy is   ", self.confusion(y,d))

            return

        def load_data(self):
            mydata = np.genfromtxt("IrisDataset.txt", delimiter=",")
            #print(mydata)
            self.my_data=mydata
        def testt(self):
            lisst=np.array([1,2,3])

            mse =(lisst ** 2)
            print (mse)


if __name__=='__main__':
    rbf_obj=RBF(17)
    #rbf_obj.testt()
    rbf_obj.train()
    rbf_obj.test()
