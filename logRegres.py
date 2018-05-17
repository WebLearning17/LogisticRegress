#-*- coding=UTF-8 -*-
#逻辑回归
import  numpy as np
import matplotlib.pyplot as plt
#梯度上升算法的具体实现

#加载数据
def loadDataSet():
    dataMat=[]
    labelMAt=[]
    fr=open("testSet.txt","r")
    for line in fr.readlines():

        attr=line.strip().split()
        dataMat.append([1.0,float(attr[0]),float(attr[1])])
        labelMAt.append(int(attr[2]))
    return  dataMat,labelMAt

def sigmod(inX):
    return  1.0/(1+np.exp(-inX))


def gradAscent(dataMat,classLabels):
    dataMatrix=np.mat(dataMat)
    labelMat=np.mat(classLabels).transpose()

    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))

    for k in range(maxCycles):
        h=sigmod(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return  weights
#画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(wei):

    if type(wei).__name__!='ndarray':
        weights=wei.getA()
    else:
        weights=wei
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')

    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

#随机梯度上升算法
def stocGradAscents0(dataMatrix,classLabels,numIter=150):
    m,n=np.shape(dataMatrix)

    weights=np.ones(n)
    dataIndex = range(m)
    for j in range(numIter):

        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            h=sigmod(sum(dataMatrix[randIndex],weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
    return weights

def classifyVector(inX,weights):
    prob=sigmod(sum(inX*weights))
    if prob>0.5:
        return 1
    else:
        return 0


#实验测试
'''
dataArr,labelMat=loadDataSet()

#梯度上升算法
#weights=gradAscent(dataArr,labelMat)
#第二种 随机梯度上升算法
weights=stocGradAscents0(np.array(dataArr),labelMat)
print weights
print type(weights).__name__

plotBestFit(weights)

'''