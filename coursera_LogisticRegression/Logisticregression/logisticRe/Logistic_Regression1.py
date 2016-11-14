__author__ = 'John'
from numpy import *
import matplotlib.pyplot as plt
import coursera_loaddata_numpy as C1
import Logistic_data_processing as C2  #vital!  Wu!la...!
#define S function; sigmoid(x)=1.0/(1+exp(-inX))
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
#gradient descent getting best weights(only using the GD method)
def Logistic_prepare(filename):
    #not scaling!
    f_not_scaling=C1.load_tonumpy(filename)
    #devide for ploting
    feature_not_scaling_0,feature_not_scaling_1=C2.devide(f_not_scaling)
    #the first column should be all 1
    m0,n0=shape(f_not_scaling)
    fprepare_not_scaling=zeros([m0,n0+1])
    fprepare_not_scaling[:,0]=1.0
    fprepare_not_scaling[:,1:]=f_not_scaling
    #scaling
    f_scaling=C2.Logistic_scaling(f_not_scaling)
    #devide for ploting
    feature_scaling_0,feature_scaling_1=C2.devide(f_scaling)
    m1,n1=shape(f_scaling)
    #the first column should be all 1
    fprepare_scaling=zeros([m1,n1+1])
    fprepare_scaling[:,0]=1.0
    fprepare_scaling[:,1:]=f_scaling
    return fprepare_not_scaling,feature_not_scaling_0,feature_not_scaling_1,fprepare_scaling,feature_scaling_0,feature_scaling_1
def gradAscent(f):
    #Logistic not scaling!
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
        weights = weights + alpha * dataMatrix.transpose()*error #matrix and Gradient Descent
    #************************************************************************
    #ending!!!!!!!!!
    return weights
#Logistic results outputing and ploting,data points and decision boundary
def Logistic_result_output(filename):
    fprepare_not_scaling,feature_not_scaling_0,feature_not_scaling_1,fprepare_scaling,feature_scaling_0,feature_scaling_1=Logistic_prepare(filename)
    m,n=shape(fprepare_scaling)
    weights_scale=gradAscent(fprepare_scaling)
    weights_not_scale=gradAscent(fprepare_not_scaling)
    #scaling results
    print("scaling results!")
    print("the fitest weights are:")
    print weights_scale
    #ploting
    #plot the data point
    ax1=plt.subplot(111)
    ax1.set_title("feature_scaling_Logisticregression")
    #y=0
    ax1.scatter(feature_scaling_0[:,0],feature_scaling_0[:,1],s=m,c='r',marker='*')
    #y=1
    ax1.scatter(feature_scaling_1[:,0],feature_scaling_1[:,1],s=m,c='b',marker='o')
    #plot the decision boundary*********
    # y=-(xowo+x1w1)/w2
    x1=arange(0,1,0.1)
    x0=1
    y=(-x0*weights_scale[0,0]-x1*weights_scale[1,0])/weights_scale[2,0]  #attention weights is matrix type
    ax1.plot(x1,y)
    plt.show()
    #origin
    #not scaling results
    #print("not scaling results!")
    #print("the fitest weights are:")
    #print weights_not_scale
    #ax2=plt.subplot(122)
    #ax2.set_title("no_feature_scaling_Logisticregression")
    ##y=0
    #ax2.scatter(feature_not_scaling_0[:,0],feature_not_scaling_0[:,1],s=m,c='r',marker='*')
    ##y=1
    #ax2.scatter(feature_not_scaling_1[:,0],feature_not_scaling_1[:,1],s=m,c='b',marker='o')
    ##plot the decision boundary*********
    ##y=-(xowo+x1w1)/w2
    #x11=arange(0,120,1)
    #x00=1
    #yy=(-x00*weights_not_scale[0,0]-x11*weights_not_scale[1,0])/weights_not_scale[2,0]  #attention weights is matrix type
    #ax2.plot(x11,yy)
    #showing!
    #plt.show()


