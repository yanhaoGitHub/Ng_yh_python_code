#-*- coding:UTF-8 -*-
import sys
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from pca import *

if __name__ == '__main__':
    fig, ax = plt.subplots(1,2,figsize=(10,5))  #figsize=(10,5)用来控制生成图片的大小
    ex7data1 = np.load('ex7data1.npz')
    x = ex7data1['X']
    m,n = x.shape
    ax[0].scatter(x[:,0],x[:,1],c='b',marker='o')
    ax[0].set_title('Original Data')
    norm,mean,std = normalize(x)
    ax[1].scatter(norm[:,0],norm[:,1],c='b',marker='*')
    ax[1].set_title('Normalized Data')
    u,s = pca(norm)  #得到特征向量和特征值
    z = project(norm,u,1)
    # recovery from projected data
    xr = recovery(z,u,1)
    ax[1].scatter(xr[:,0],xr[:,1],c='r',marker='+')

    # reverse operation of normalization on approximate reconstruction xr
    xrr = revernorm(xr,mean,std)  #标准化的逆过程
    ax[0].scatter(xrr[:,0],xrr[:,1],c='r',marker='*')
    
    for i in range(0,m):
        print(x[i])
        line0 = np.vstack((x[i],xrr[i]))
        print(line0)
        line1 = np.vstack((norm[i],xr[i]))
        ax[0].plot(line0[:,0],line0[:,1],'k--')
        ax[1].plot(line1[:,0],line1[:,1],'k--')

    plt.show()
