import numpy as np
import pandas as pd


class GHA:
    df = pd.read_fwf("5amis_test")
    df2 = pd.read_fwf("5amis_x")
    x = df2.as_matrix()
    test = df.as_matrix()
    pca_features = 100
    np.random.seed(0)
    w = np.random.randn(pca_features, 2500) * 0.01

    def forward (self, w, data):
        z = np.dot(w, data.T)
        return z

    def update_w(self, net, lr):
        new_w = ((np.dot(net, self.x)) - (np.dot(net, np.dot(self.w.T, net).T))) * lr
        return new_w

    def train(self, lr, epoch):
        for i in range(epoch):
            net = self.forward(self.w, self.x)
            new_w = self.update_w(net, lr)
            self.w += new_w * lr


if __name__ == '__main__':
    H = GHA()
    H.train(.0000001, 400)
    x = H.forward(H.x, H.w)
    test = H.forward(H.test, H.w)
    np.savetxt("new2_test", test, delimiter='   ', fmt='%f')
    np.savetxt("new2", x, delimiter='   ', fmt='%f')


