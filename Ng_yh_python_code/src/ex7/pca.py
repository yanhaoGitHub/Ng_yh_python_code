#-*- coding:UTF-8 -*_
import numpy as np
from numpy.linalg import svd
import scipy.optimize as op
import matplotlib.pyplot as plt
def normalize(x):   #归一化
    """
    returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    This is often a good preprocessing step to do when working with learning algorithms.
    """
    mean = np.mean(x,0)
    std = np.std(x,0)
    norm = (x - mean)/std
    return norm,mean,std

def revernorm(x,mean,std):  #逆归一化，上一个方法的逆过程
    """
    The reverse operation of nomalization
    """
    return mean + (x * std)

def pca(x):
    """
    computes eigenvectors of the covariance matrix of X
    Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """
    m,n = x.shape
    Sigma = 1.0/m * x.T.dot(x)
    u,s,v = svd(Sigma)  #奇异值分解
    return u,s   #返回特征向量u和特征值s

def project(x,u,k):  #投射，将x投射到特征向量u的前k列表示的低维空间中
    """
    computes the projection of  the normalized inputs X into the reduced dimensional space-
    spanned by the first K columns of U. It returns the projected examples.
    """
    return x.dot(u[:,0:k])

def recovery(z,u,k):  #还原，将z矩阵还原为投射之前的矩阵
    """
    recovers an approximation the  original data that has been reduced to K dimensions. 
    It returns the approximate reconstruction.
    """
    return z.dot(u[:,0:k].T)
