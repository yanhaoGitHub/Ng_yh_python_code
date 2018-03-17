#-*- coding:UTF-8 -*-
import numpy as np
import Testpandas as pd
import scipy.optimize as op
import matplotlib
import matplotlib.pyplot as plt
from optimlog import *
'''
    this module is the LR of the first scatter plot, finally plot a line to seperate two kinds dots.
'''
def plotScatter(ax,data):
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='Admitted')  #plot postive data
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='Not admitted')  #plot negative data
    ax.set_xlabel('Exame 1 score')
    ax.set_ylabel('Exame 2 score')
    ax.legend(loc='best')

def plotBoundary(ax,theta,x):
    xmax = np.max(x[:,1])
    xmin = np.min(x[:,1])
    score1 = np.arange(xmin,xmax,0.1)
    #下面计算score2的公式是根据theta[1]*x[1] + theta[2]*x[2] + theta[0]*x[0] = 0推导出来的
    #那么为什么上面的式子要等于0呢？因为z=0是决策边界，z>0,即可判定是1，z<0即可判定为0
    score2 = (-1.0/theta[2])*(theta[0]+theta[1]*score1)
    ax.plot(score1,score2,label='Boundary')
    ax.legend(loc='best')

if __name__ == '__main__':
    fig, ax = plt.subplots()
    data = np.loadtxt('ex2data1.txt',delimiter=',')
    #下面代码解释：np.delete用来删除data中维度为1（维度为0表示行，为1表示列）的最后一列（-1）数据
    x = np.delete(data,-1,axis=1)
    #print("before x="+str(x))
    #np.hstack就是按列顺序把数组堆叠起来，vstack和它正好相反
    x = np.hstack((np.ones((x.shape[0],1)),x))
    #print("later x="+str(x))
    y = data[:,-1]

    m,n = x.shape
    #print(x.shape)  #(100, 3)
    plotScatter(ax,data)
    #初始化theta，全部为0
    theta = np.zeros(n)
    #调用库函数迭代优化theta
    res = optimSolve(theta,x,y)
    #画出决策边界
    plotBoundary(ax,res,x)
    plt.show()
