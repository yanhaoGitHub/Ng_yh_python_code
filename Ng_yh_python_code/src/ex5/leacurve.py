#-*- coding:UTF-8 -*-
import sys
sys.path.append('..')
import numpy as np
from linear import *

def addOnes(x,m):
    """
    Add a col of 1 at first column of x
    x: the number of rows
    """
    n = x.size/m
    one = np.ones((m,1))
    x = x.reshape((m,n))
    judge = np.sum(x[:,0] == one.flatten())
    if judge != m:
        x = np.hstack((one,x))
    return x

def leacurve(x,y,xval,yval,lamda):
    """
    this function implements code to generate the learning curves that will be useful in debugging learning algorithms.
    To plot the learning curve, we need a training and cross validation set error for different training set sizes.
    To obtain different training set sizes,the function uses different subsets of the original training set x.
    Specifically, for a training set size of i, you should use the first i examples (i.e., x(0:i) and y(0:i))
    """
    m1 = y.size  #得到训练集y
    m2= yval.size  #得到交叉验证集yval
    x = addOnes(x,m1)   #给训练集x中的第一列加一列1，加1的个数是m1
    xval = addOnes(xval,m2)  #给交叉验证集xval中的第一列加一列1，加1的个数是m2

    m,n = x.shape  #得到训练集的行数和列数
    errtrain = np.zeros(m)  #训练集误差
    errval = np.zeros(m)  #交叉验证集误差

    thetaInit = np.zeros(n)  #初始化theta值
    for i in range(1,m+1):
        #这里需要注意，学习曲线是训练数据集数据个数与训练集以及交叉验证集误差的函数，在计算误差的时候，交叉验证集的全部数据都参与计算，而训练集的数目是逐渐增多
        status,theta = optimSolve(thetaInit,x[0:i],y[0:i],reg=True,lamda=lamda)
        errtrain[i-1] = costFunc(theta,x[0:i],y[0:i])
        errval[i-1] = costFunc(theta,xval,yval)
    return errtrain, errval

def errAndLamda(x,y,xval,yval,lamdas):
    m1 = y.size
    m2= yval.size
    x = addOnes(x,m1)
    xval = addOnes(xval,m2)
    nlam = lamdas.size
    errtrain = np.zeros(nlam)
    errval = np.zeros(nlam)
    thetaInit = np.zeros(np.size(x,1))
    i = 0
    for lam in lamdas:
        status,theta = optimSolve(thetaInit,x,y,reg=True,lamda=lam)
        errtrain[i] = costFunc(theta,x,y)
        errval[i] = costFunc(theta,xval,yval)
        i += 1
    return errtrain, errval
        

"""
if __name__ == '__main__':
    x = np.loadtxt('x.txt')
    y = np.loadtxt('y.txt')
    xval = np.loadtxt('xval.txt')
    yval = np.loadtxt('yval.txt')
    errtrain,errval =  leacurve(x,y,xval,yval,0.0)
"""
