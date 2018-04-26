import math
import numpy as np
import  copy
import pandas as pd



class RBF :
        def __init__(self,k):
            self.k=k
            self.load_data()
            self.hidden = np.zeros(len(self.kmean(k,self.my_data)))
            print(self.hidden)

        def segma(self): # ?
            d=1
            #Max. distance between any 2 centroids
            self.sigma= d / math.sqrt(2 * self.k)

        def kmean(self,k,data_set):
            # comment this and add h ,w dimensions
            #k=4;

            h, w = 10, 2;
            w=len(data_set[0])
            h=len(data_set)
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

        def testt(self):
            lisst=[[1,2,3],[]]

            l= copy.deepcopy(lisst)
            l[0].append(4)
            print(l)
            print(lisst)
        def train(self):
            print ("train")
        def test(self):
            print ("test")
        def load_data(self):
            mydata = np.genfromtxt("IrisDataset.txt", delimiter=",")
            #print(mydata)
            self.my_data=mydata
if __name__=='__main__':
    rbf_obj=RBF(5)
    #rbf_obj.testt()
