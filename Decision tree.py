from numpy import *
from sklearn import tree
def file2metrix(filename):
    fr=open(filename)#打开文件
    numline=fr.readlines()#读取文件的行向量
    m=len(numline)#计算行向量的个数
    returnmat=zeros((m,460))#初始化要返回的特征数组
    classlabel=[]
    index=0
    for line in numline:
        line=line.strip( )
        listfoemline=line.split('\t')
        returnmat[index,:]=listfoemline[0:460]
        classlabel.append(int(listfoemline[-1]))
        index+=1
    return returnmat,classlabel
'''将文件数据转化为测试数据和训练数据'''
def get_trian_testdata(filename):
    dataMat,dataClassLabel=file2metrix(filename)
    return dataMat,dataClassLabel
'''
   max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples（default=2）.
        Ignored if ``max_leaf_nodes`` is not None.
'''
'''主函数'''
if __name__ == "__main__":
    trainMat,trainClassLabel=get_trian_testdata('adpaTr.txt')
    testMat,testClassLabel=get_trian_testdata('adpaTe.txt')
    print ('训练数据集为：',len(trainClassLabel))
    print ('测试数据集为：',len(testClassLabel))
    '''max_depth没被设置'''
    clf=tree.DecisionTreeClassifier()
    clf=clf.fit(trainMat,trainClassLabel)
    testAcc=clf.score(testMat,testClassLabel)
    print ('决策树的识别率：', testAcc * 100, '%')