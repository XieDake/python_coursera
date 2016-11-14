__author__ = 'John'
#import the needing package!
from numpy import *
import coursera_loaddata_numpy as C1
import Logistic_data_processing as C2
import matplotlib.pyplot as plt
#define S function; sigmoid(x)=1.0/(1+exp(-inX))
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
def Logistic_prepare(filename):
    #not scaling!
    f_not_scaling=C1.load_tonumpy(filename)
    #devide for ploting
    feature_not_scaling_0,feature_not_scaling_1=C2.devide(f_not_scaling)
    #feature_mapping
    f_mapping=C2.feature_mapping(f_not_scaling)
    #the first column should be all 1
    m,n=shape(f_mapping)
    fprepare_not_scaling=zeros([m,n+1])
    fprepare_not_scaling[:,0]=1.0
    fprepare_not_scaling[:,1:]=f_mapping
    ##scaling
    #f_scaling=C2.Logistic_scaling(f_not_scaling)
    ##devide for ploting
    #feature_scaling_0,feature_scaling_1=C2.devide(f_scaling)
    #m1,n1=shape(f_scaling)
    ##the first column should be all 1
    #fprepare_scaling=zeros([m1,n1+1])
    #fprepare_scaling[:,-1]=1.0
    #fprepare_scaling[:,0:-1]=f_scaling
    return fprepare_not_scaling,feature_not_scaling_0,feature_not_scaling_1
#Regularization_LogisticRegression
def gradAscent(f,namda):
    #Logistic not scaling!
    namda=namda
    classlables=f[:,-1]
    Xfeatures=f[:,0:-1]
    dataMatrix = mat(Xfeatures)             #convert to NumPy matrix
    labelMat = mat(classlables).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 1000
    weights = ones((n,1))
    for k_not_scale in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights - alpha * dataMatrix.transpose()*error #weight[0] Gradient Descent
        weithts_0=weights
        weights = weights*(1-alpha*namda) - alpha * dataMatrix.transpose()*error#weight[1:] Gradient Descent
        weights_1=weights
        weights[0]=weithts_0[0]
        weights[1:]=weights_1[1:]
    #************************************************************************
    #ending!!!!!!!!!
    return weights
def Regular_LogR_result_output(filename,namda):
    fprepare_not_scaling,feature_not_scaling_0,feature_not_scaling_1=Logistic_prepare(filename)
    namda=namda
    m,n=shape(fprepare_not_scaling)
    #not need to scale the features
    weights=gradAscent(fprepare_not_scaling,namda)
    #weights outputing!
    print("the fitest weights are:")
    print weights
    #ploting
    #ploting the data points!
    ax1=plt.subplot(111)
    ax1.set_title("Regularization_Logisticregression")
    #y=0
    ax1.scatter(feature_not_scaling_0[:,0],feature_not_scaling_0[:,1],s=m,c='r',marker='*')
    #y=1
    ax1.scatter(feature_not_scaling_1[:,0],feature_not_scaling_1[:,1],s=m,c='b',marker='o')
    #plot the decision boundary*********
    #w0+w1x1+w2x2+w3x1^2+w4x1x2+w5x2^2
    X1=arange(-1,1,0.1)
    X2=arange(-1,1,0.1)
    x1,x2=meshgrid(X1,X2)
    ax1.contour(x1,x2,weights[0,0]+weights[1,0]*x1+weights[2,0]*x2+weights[3,0]*x1**2+weights[4,0]**x1*x2+weights[5,0]*x2**2 ,0)
    plt.show()






