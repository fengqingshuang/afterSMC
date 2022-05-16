# 一切按专利里的参数在做
import time
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import tools as tools
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import xlsxwriter
import random
import pywt

class lightweightPilae(object):
    def __init__(self, p, k):
        self.p = p
        self.k = k
        # 用于存储变量的list
        self.we = 0
        self.softmaxTrainAcc = []
        self.softmaxTestAcc = []


    def svdfree(self, inputX):
        t1 = time.time()
        dim = inputX.shape[0]
        k = self.k
        p = self.p
        N = inputX.shape[1]
        H0 = np.hstack((np.eye(p), np.zeros((p, N-p))))
        we0 = H0.dot(inputX.T)
        # print(time.time()-t1,'得到we0的时间')
        # 下面计进行权重归一化
        xxt = inputX.dot(inputX.T)
        we = self.norm(we0, xxt, qr=0)
        # print("train lightweight PIL cost time {:.5f}s".format(time.time() - t1))
        return we

    def norm(self, we0, xxt, qr):
        if qr ==0:
            U, value, VT = np.linalg.svd(xxt, full_matrices=0)
        elif qr == 1:
            v = np.eye(xxt.shape[0])
            value, U = self.qr_iteration(xxt, v)
        else:
            print('选择是否使用qr分解计算特征值和特征向量')
        value.flags.writeable = True
        # 这里的参数可以调整！！
        value[value < 1e-6] = 0
        value[value != 0] = 1 / value[value != 0]
        b_tem = np.diag(value)  # 由特征值倒数组成的d*d的矩阵
        right = U.dot(b_tem).dot(U.T)  # 右边的三项相乘
        we = we0.dot(right)
        return we

    # QR分解算法 返回特征值和特征向量
    def qr_iteration(self, A, v):
        for i in range(100):
            Q, R = np.linalg.qr(A)
            v = np.dot(v, Q)
            A = np.dot(R, Q)
        return np.diag(A), v

    def activeFunc(self, tempH, epsilon=0.0008):
        tempH[tempH <= epsilon] = 0
        tempH[tempH > epsilon] = 1
        return tempH

    # 这一步的目的是获取权重，然后调用分类器
    # 训练的时候把训练集和测试集的x都放进去，保证同分布
    def trainSvdfree(self, trainX):
        we = self.svdfree(trainX)
        self.we = we
        # hTem = we.dot(trainX)
        # diag_plot(hTem)
        # aTem = get_offDiag(hTem)
        # offDiag_plot(aTem)
        # H = self.activeFunc(hTem)

    def classifier(self, trainX, trainY, testX, testY):
        trainX = self.activeFunc(self.we.dot(trainX))
        testX = self.activeFunc(self.we.dot(testX))
        # trainX = self.we.dot(trainX)
        # testX = self.we.dot(testX)
        train_acc, test_acc = self.predict_softmax(trainX.T, trainY, testX.T, testY)
        return train_acc, test_acc
        # trainX.T是N*d的矩阵

    # 训练逻辑回归
    def train_softmax(self, train_X, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)
        model.fit(train_X, train_y)
        return model

    # 测试逻辑回归
    def predict_softmax(self, train_X, train_y, test_X, test_y):
        model = self.train_softmax(train_X, train_y)
        train_y_predict = model.predict(train_X)
        test_y_predict = model.predict(test_X)
        train_acc = accuracy_score(train_y, train_y_predict) * 100
        test_acc = accuracy_score(test_y, test_y_predict) * 100
        self.softmaxTrainAcc.append(train_acc)
        self.softmaxTestAcc.append(test_acc)
        # print("Softmax Train accuracy:{}% | Test accuracy:{}%".format(train_acc, test_acc))
        return train_acc, test_acc
        # test_recall_score = recall_score(test_y, test_y_predict, average='micro') * 100
        # test_f1_score = f1_score(test_y, test_y_predict, average='micro') * 100
        # test_classification_report = classification_report(test_y, test_y_predict)
        # print("test recall:{}, f1_score:{}".format(test_recall_score, test_f1_score))
        # print(self.test_classification_report)

'''取出非对角线值并画直方图'''
def get_offDiag(e_tem):
    a_tem=[] # 存非对角线所有值
    for i in range(len(e_tem)):
        for j in range(len(e_tem[0])):
            if i!=j:
                a_tem.append(e_tem[i][j])
    return a_tem

def offDiag_plot(a_tem):
    a_tem = random.sample(a_tem, 1000)
    non_bins = np.linspace(min(a_tem), max(a_tem), 300)  # 设置范围和分段
    plt.hist(a_tem, non_bins)
    x_major_locator = MultipleLocator(1)  # x标度
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  # 设置x标度
    plt.show()

'''取出对角线值并画直方图'''
def diag_plot(e_tem):
    diag_e = np.diag(e_tem)  # 取对角线
    print('对角线长度',len(diag_e))
    bins = np.linspace(min(diag_e),max(diag_e), 50)
    plt.hist(diag_e, bins)
    x_major_locator2 = MultipleLocator(1)
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(x_major_locator2)
    plt.show()

def haar_wavelet(X_all,d):
    N = len(X_all)
    d_haar = int(d/4)
    X_all_haar = np.zeros([N, d_haar])
    for i in range(N):
        LL1, coeffs2 = pywt.dwt2(X_all[i], 'haar')
        X_all_haar[i] = LL1.reshape(-1, d_haar).astype('float64') / 255.
    return X_all_haar

if __name__ == '__main__':
    k = 0.8  # 正则化系数 在提取特征值时好像消去了，在最后一层的解码器有用
    p = 290  # 第一层隐层神经元数
    d = 784
    # 若是mnist，特征值为784，白化除于255
    (X_all, y_all), (X_test, y_test) = tools.load_npz(r'./dataset/mnist/mnist.npz')
    # X = np.vstack((X_train, X_test))
    # y = np.vstack((y_train, y_test))
    t_haar_0 = time.time()
    X_all = haar_wavelet(X_all, d)
    t_haar = time.time() - t_haar_0
    trainAcc = []
    valAcc = []
    t = []
    for i in range(3):
        X_train, y_train, X_val, y_val = tools.split_dataset(X_all, y_all, 0.19)
        X_train = X_train.T
        X_val = X_val.T
        myPilae = lightweightPilae(p=p, k=k)
        t1 = time.time()
        myPilae.trainSvdfree(X_train)
        trainTime = time.time()-t1
        t.append(trainTime)
        # train_acc, test_acc = myPilae.classifier(X_train, y_train, X_test, y_test)
        train_acc, val_acc = myPilae.classifier(X_train, y_train, X_val, y_val)
        # print('此时的p:',p)
        print('train_acc:',train_acc)
        print('test_acc:', val_acc)
        print('time:', trainTime)
        trainAcc.append(train_acc)
        valAcc.append(val_acc)
    print('train acc:',np.mean(trainAcc),'+-',np.std(trainAcc, ddof=1))
    print('test acc:', np.mean(valAcc),'+-', np.std(valAcc, ddof=1))
    print('average time',np.mean(t), '+-',np.std(t))
    # workbook = xlsxwriter.Workbook("svdfree_mnist_acc.xlsx")
    # worksheet = workbook.add_worksheet()
    # for i in range(len(trainAcc)):
    #     # worksheet.write(i, 0, plus*i+begin)
    #     worksheet.write(i, 1, trainAcc[i])
    #     worksheet.write(i, 2, val_acc[i])
    # workbook.close()