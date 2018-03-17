#-*- coding:UTF-8 -*-
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def sigmoid(z):
    """sigmoid(z)
    z: It can be a array, a narray, a vector and a matrix
    """
    return 1.0 / (1.0 + np.exp(-z))

def costFunction(theta):
    y1 = np.array(sigmoid(x*theta))
    y0 = np.array(y)
    cost = np.sum(y0 * np.log(y1) + (1.0 - y0) * np.log(1.0 - y1))/m
    return cost

def costDer(theta):
    """
    compute the derivative of cost funtion
    """
    h = np.array(sigmoid(x*theta))
    error = h - y
    deri = (x.T * error)/m
    return deri

def decent(alpha,theta,iters):
    """
    Use gradient decent method to compute parameters
    """
    while iters > 0:
        deri = costDer(theta)
        theta = theta - alpha*deri
        iters -= 1
    return theta

def plotScatter(ax):
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='Admitted')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='Not admitted')
    ax.set_xlabel('Exame 1 score')
    ax.set_ylabel('Exame 2 score')
    ax.legend(loc='best')

def plotBoundary(ax,theta):
    xmax = np.max(x[:,1])
    xmin = np.min(x[:,1])
    #print("theta="+str(theta)) #theta=[[-235.0402135 ],[   1.8343292 ],[   1.81370475]]
    theta = np.array(theta.T)[0]
    #print("theta2="+str(theta)) #theta2=[-235.0402135     1.8343292     1.81370475]
    score1 = np.arange(xmin,xmax,0.1)
    score2 = (-1.0/theta[2])*(theta[0]+theta[1]*score1)
    ax.plot(score1,score2,label='Boundary')
    ax.legend(loc='best')

def predict(x0,theta):
    x0 = np.hstack(([1],x0)) #就是水平(按列顺序)把数组给堆叠起来
    #a=[[1],[2],[3]]
    #b=[[1],[2],[3]]
    #c=[[1],[2],[3]]
    #d=[[1],[2],[3]]
    #print(np.hstack((a,b,c,d)))
    x0 = np.mat(x0)
    #print("x0="+str(x0))  #x0为行向量，x0=[[ 1 45 85]]
    #print("theta"+str(theta))  #列向量，theta[[-235.0402135][1.8343292 ][1.81370475]]
    #print("acacacacac=="+str(x0*theta))#在计算array数据类型时，*和multiply都表示对应元素的数乘。在计算matrix数据类型时，*就是和线性代数中矩阵运算规则一样，multiply依旧是对应元素的数乘。
    #print ("the probability is %.3f" % (sigmoid(x0*theta))) #所以这里x0和theta的*运算其实是矩阵运算！
    h = sigmoid(x0*theta)
    if h > 0.5:
        return 1.0
    else:
        return 0.0

if __name__ == '__main__':
    data = np.loadtxt('ex2data1.txt',delimiter=',')
    print("data.shape="+str(data.shape)) #(100,3)
    x = np.mat(np.delete(data,-1,axis=1)) #(100,2)
    print("x before"+str(x.shape))
    x = np.hstack((np.ones((x.shape[0],1)) , x))  #(100,3), 经过此步骤之后，在x的第一列加了一列1
    print("after x"+str(x.shape))

    #y = np.mat(data[:,-1])
    #print("y.shape="+str(y.shape)) #(1,100),这里为什么截取的是一列，但是shape是一个行向量呢？

    y = np.mat(data[:,-1]).T  #(100,1)此时y是一个列向量
    print("y="+str(y.shape))
    m,n = x.shape
    print("x.shape[0]="+str(m))
    print("x.shape[1]="+str(n))
    alpha = 0.043
    theta = np.mat(np.zeros((n,1)))
    iters = 1000000   #1000000次之后的结果正常，而10000次的结果很差
    fig, ax = plt.subplots()
    plotScatter(ax)
    theta = decent(alpha,theta,iters)
    print theta
    print predict([45,85],theta)
    plotBoundary(ax,theta)
    #print costFunction(x,y,theta)
    plt.show()

