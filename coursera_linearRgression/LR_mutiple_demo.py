__author__ = 'John'
#attentioon please:here when the dimention of the feature
#are more than 3(>=3),
#means we can not ploting for viewing
from sklearn import linear_model  #first sklearn_____linear_model; then Linear_model_____LinearRegression
from numpy import *
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
