from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 50
def file2metrix(filename):
    fr=open(filename)#打开文件
    numline=fr.readlines()#读取文件的行向量
    m=len(numline)#计算行向量的个数
    returnmat=zeros((m,24))#初始化要返回的特征数组
    classlabel=[]
    index=0
    for line in numline:
        line=line.strip( )
        listfoemline=line.split('\t')
        returnmat[index,:]=listfoemline[0:24]
        classlabel.append(int(listfoemline[-1]))
        index+=1
    return returnmat,classlabel

X,y = file2metrix("bqallPCPseAAC.txt")

classifiers = [
    ("SGD", SGDClassifier(max_iter=1000)),
    ("ASGD", SGDClassifier(average=True)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',max_iter=1000,
                                                         C=1.0)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',max_iter=1000,
                                                          C=1.0)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
]

xx = 1. - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()