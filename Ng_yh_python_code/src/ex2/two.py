#-*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimlog import *

def expandX(x,exp):
    """
    x: the input data(a matrix)
    exp: the max index of x. (x1^i * x2^j, i + j = exp)
    """
    row,col = x.shape
    #单独提取出x的第2列，即x1
    param1 = x[:,1].reshape((row,1))
    #单独提取出x的第3列，即x2
    param2 = x[:,2].reshape((row,1))
    for index in np.arange(2,exp+1):
        for i in np.arange(0,index+1):
            #求param1的index-i次方
            tmp1 = np.array(np.power(param1,index - i))
            #求param2的i次方
            tmp2 = np.array(np.power(param2,i))
            x = np.hstack((x,tmp1*tmp2))
    return x

def plotScatter(ax,data):
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='y = 1')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='y = 0')
    ax.set_xlabel('Microchip Test1')
    ax.set_ylabel('Microchip Test2')
    ax.legend(loc='best')

def plotBoundary(ax,x,theta):
    x1 = np.linspace(-1,1.5,50) #确定x1的边界，分为50份
    x2 = np.linspace(-1,1.5,50) #确定x2的边界，分为50份
    z = np.zeros((np.size(x1),np.size(x2)))  #50,50
    tmp = np.ones((1,3))
    for i in np.arange(0,np.size(x1)):
        for j in np.arange(0,np.size(x2)):
            tmp[0,1] = x1[i]
            tmp[0,2] = x2[j]
            ex = expandX(tmp,6)  #将tmp扩展为6次多项式
            z[i,j] = ex.dot(theta)
    x1,x2 = np.meshgrid(x1,x2)
    print("x1.shape="+str(x1.shape)) #（50,50）
    print("x2.shape="+str(x2.shape)) #（50,50）
    print("x1="+str(x1))
    print("x2="+str(x2))
    ax.contour(x1,x2,z.T,[0],label='Boundary')
    ax.legend(loc='best')

def predict(x,theta):
    x = np.hstack(([1],x))
    x = np.mat(x)
    x = expandX(x,6)
    print ("the probability is %.3f" % (sigmoid(x*theta)))
    h = np.sum(sigmoid(x*theta))
    if h > 0.5:
        return 1.0
    else:
        return 0.0
    
if __name__ == '__main__':
    index = 6
    alpha = 0.9
    iters = 5000
    lamda = 1
    fig, ax = plt.subplots()
    data = np.loadtxt('ex2data2.txt',delimiter=',')
    #调用函数，画出所有点
    plotScatter(ax,data)

    #删掉data的最后一列，即删掉y值所在的那一列
    x = np.delete(data,-1,axis=1)
    #在x第0列加一列全1的数据
    x = np.hstack((np.ones((x.shape[0],1)),x))
    #进行多项式特征扩充
    x = expandX(x,index)
    print(x.shape)  #(118, 28)
    y = data[:,-1]

    row,col = x.shape
    # initialize theta with zero vector
    theta = np.ones(col)
    #predict([0.25,1.5],res)
    for lamda in [0,1,100]:
        res = optimSolve(theta,x,y,reg=True,lamda=lamda)
        #对每个lamda都画一次边界
        plotBoundary(ax,x,res)

    plt.show()
