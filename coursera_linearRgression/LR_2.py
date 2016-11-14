# -*- coding: utf-8 -*-
__author__ = 'John'
#mutiple varibles regression(>=2)
# -*- coding: utf-8 -*-
#LR needs thoese libs
from sklearn import linear_model  #first sklearn_____linear_model; then Linear_model_____LinearRegression
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import coursera_loaddata_numpy as C1
def coursera_LR(filename):
    f_LR=C1.load_tonumpy(filename)
    m,n=shape(f_LR)
    #get X and the lable Y,
    #attention please the features X, first column is all 1,so we need to make change to fit this
    X_feature=zeros([m,n])
    X_feature[:,0]=1
    X_feature[:,1:-1]=f_LR[:,0:-2]
    y_lable=f_LR[:,-1]
    #get LR model from sklearn.linear model
    LR=linear_model.LinearRegression()
    #training the LR model to get the weights of the LR
    LR.fit(X_feature,y_lable)
    weight_LR=LR.coef_#coef_ is weight of the LR
    print("the training weights of the LR model is:" )
    print weight_LR
    print LR.predict
    #ploting for better viewing
    #ploting together:scatter and plot_surface
    ax1=plt.subplot(111,projection='3d')  #build 3d project
    ax1.scatter(X_feature[:,1],X_feature[:,2],y_lable)
    #second subplot,showing the surface of the regression
    #ax2=plt.subplot(111,projection='3d')  #build 3d project
    X=arange(500.0,5000.0,500.0)
    Y=arange(0.0,9.0,1.0)
    X,Y=meshgrid(X,Y)
    Z=X*weight_LR[1]+Y*weight_LR[2]
    ax1.plot_surface(X,Y,Z)
    #showing
    plt.show()
