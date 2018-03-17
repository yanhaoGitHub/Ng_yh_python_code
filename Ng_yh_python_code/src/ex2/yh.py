#-*- coding:UTF-8 -*-
import numpy as np
import Testpandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def cost(theta):   #下次从这里继续写！
    hx = np.array(sigmoid(x*theta))
    y0 = np.array(y)
    cost = np.sum(y0*np.log(hx)+(1.0-y0)*np.log(1.0-hx))/m
    return cost

def costDer(theta):
    hx = np.array(sigmoid(x*theta))
    error = hx - y
    deri = (x.T * error)/m
    return deri   #Why is x.T?

def decent(alpha, theta, iters):
    while iters>0:
        deri = costDer(theta)
        theta = theta- alpha*deri
        iters -= 1
    return theta

def plotScatter(ax):
    pos = data[data[:,2] == 1]
    nag = data[data[:,2] == 0]
    ax.scatter(pos[:,0], pos[:,1], c="r", marker='+', label='admitted')
    ax.scatter(nag[:,0], nag[:,1], c="g", marker='o', label='noadmitted')
    ax.set_xlabel("exam1")
    ax.set_ylabel('exam2')
    ax.legend(loc='best')

def plotBoundary(ax, theta):
    min = np.min(x[:,1])
    max = np.max(x[:,1])
    theta = np.array(theta.T)[0]
    x1 = np.arange(min,max,0.5)

    x2 = (-1.0/theta[2])*(theta[0]+theta[1]*x1)
    ax.plot(x1,x2,c='r',label="boundary")
    ax.legend(loc='best')

if __name__ == '__main__':
    #data = pd.read_csv('ex2data1.txt',header=None,names=['exam1','exam2','admitted'])
    #采用上面的读入数据方式，数据读进来是一行一行的，所以采用矩阵操作会报错，以后就采用下面这种写法！
    data = np.loadtxt('ex2data1.txt',delimiter=',')  #数据读进来的时候是矩阵
    print("data.shape="+str(data.shape))
    x = np.mat(np.delete(data,-1,axis=1))
    x = np.hstack((np.ones((x.shape[0],1)), x))  #注意这里面坑太多了，少个括号都不行！np.ones()中还有个括号，别忘记！！
    y = np.mat(data[:,-1]).T  #这里曾经忘记了转置，导致最后的决策边界很差！！！

    m,n = x.shape
    alpha = 0.043
    theta = np.zeros((n,1))
    iters = 1000000

    theta = decent(alpha,theta,iters)
    print(theta)

    fig, ax = plt.subplots()
    plotScatter(ax)
    plotBoundary(ax,theta)
    plt.show()


