#!/usr/bin/python

#Exercise 1: Linear Regression

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def costFunc(x,y,theta):
    """costFunc(x,y,theta)
    x: is a  coefficient matrix
    y: the result matrix (result vector)
    theta: the learning rate
    """
    m = x.shape[0]
    tmp = x*theta - y
    return ((tmp.T * tmp)/(2.0*m))[0,0]

"""
# this is recursive implementation
def gradient(x,y,theta,alpha,iterations):
    if iterations < 0:
        return theta
    else:
        down =  ((x*theta - y).T * x)
        down = (alpha/x.shape[0]) * down.T
        theta = theta - down
        return gradient(x,y,theta,alpha,iterations-1)
"""

# iteration implementation
def gradient(x,y,theta,alpha,iterations):
    jhistory = []
    while iterations > 0:
        down =  ((x*theta - y).T * x)
        down = (alpha/x.shape[0]) * down.T
        theta = theta - down
        iterations -= 1
        jhistory.append(costFunc(x,y,theta))
    return theta,jhistory

# pridict the the value for the given x, according to the calculated  theta
def predict(theta,x):
    """
    x: the input value
    theta:   calculated parameters matrix
    """
    return np.mat([1,x]) * theta

# plot the 3D's cost function 
def visualCost3D(x,y):
    """visualCost3D(cost,xrange,yrange):
    cost: the cost function you will call
    xrange: xrange[0]: the start point; xrange[1]: the end point; xrange[2]: the number of data
    yrange: yrange[0]: the start point; yrange[1]: the end point; yrange[2]: the number of data
    """
    theta0 = np.linspace(-10,10,200)
    theta1 = np.linspace(-1,4,200)
    jvals = np.zeros((theta0.size,theta1.size))
    for i in np.arange(0,theta0.size):
        for j in np.arange(0,theta1.size):
            theta = np.mat([theta0[i],theta1[j]]).T
            jvals[i,j] = costFunc(x,y,theta)
    fig = plt.figure()
    ax = Axes3D(fig)
    x,y = np.meshgrid(theta0,theta1)
    ax.plot_surface(x,y,jvals,cmap='rainbow')
    ax.set_title('Cost Function Surface')

def contour(x,y,res):
    theta0 = np.linspace(-10,10,100)
    theta1 = np.linspace(-1,4,100)
    jvals = np.zeros((theta0.size,theta1.size))
    for i in np.arange(0,theta0.size):
        for j in np.arange(0,theta1.size):
            theta = np.mat([theta0[i],theta1[j]]).T
            jvals[i,j] = costFunc(x,y,theta)
    fig,ax = plt.subplots()
    x,y = np.meshgrid(theta0,theta1)
    ax.scatter(res[0,0],res[1,0])
    ax.contour(x,y,jvals,np.logspace(-2,3,50))
    ax.set_title('Contour of Cost Function')
    

if __name__ == '__main__':
    iterations = 1500
    theta = np.mat([0,0]).T
    alpha = 0.01
    data = np.loadtxt('ex1data1.txt',delimiter=',') 
    x = data[:,0]
    y = data[:,1]

    # scatterplot of original data
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].scatter(x,y)

    #mx = np.mat(np.c_[np.ones((data.shape[0],1)),x]) 
    mx = np.hstack((np.ones((data.shape[0],1)),np.mat(x).T))
    my = np.mat(y).T
    print costFunc(mx,my,theta)
    theta,j = gradient(mx,my,theta,alpha,iterations)
    print theta
    # plot the  predict values
    ax[0].plot(mx[:,1],mx*theta)
    ax[0].set_title('Original scatterplot and Predict line')
    
    # plot the cost curve
    ax[1].plot(np.arange(1,iterations+1),j)
    ax[1].set_title('The the curve between cost value  and iteration times')
    #print predict(theta,3500)
    visualCost3D(mx,my)
    contour(mx,my,theta)
    plt.show()
