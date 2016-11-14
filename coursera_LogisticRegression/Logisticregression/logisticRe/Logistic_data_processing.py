__author__ = 'John'
import coursera_loaddata_numpy as C1  # my loading programming!
from numpy import *
#plot 2D figure,,not 3D,so only plt is enough!
import matplotlib.pyplot as plt
#feature mapping(using in the non-linear classification)
#extend x1,x2 to 5 dimention.:[x1,x2,x1^2,x1x2,x2^2,y_lable],also the lable dimention
def feature_mapping(f):#f=[x1,x2,y_lable]
    m,n=shape(f)
    f_mapping=zeros([m,6])
    f_mapping[:,0]=f[:,0] #x1
    f_mapping[:,1]=f[:,1] #x2
    f_mapping[:,2]=f[:,0]*f[:,0] #x1^2
    f_mapping[:,3]=f[:,0]*f[:,1] #x1x2
    f_mapping[:,4]=f[:,1]*f[:,1] #x2^2
    f_mapping[:,5]=f[:,-1] #y_lable
    #yes,of course!you can extend it to a more complicated form!
    return f_mapping
def devide(f):
    f=f
    #classify into two class by lable
    m,n=shape(f)
    XY_0=zeros([m,n-1])
    XY_1=zeros([m,n-1])
    #cycle controling
    x0=0
    x1=0
    for num in range(m):
        if(f[num,2]==0.0):
            XY_0[x0]=f[num,0:n-1]
            x0=x0+1
        else:
            XY_1[x1]=f[num,0:n-1]
            x1=x1+1
    # moving the (0,0) from the data set
    c=0
    d=0
    for q1 in XY_0:
        if(q1[0]!=0.0)and(q1[-1]!=0.0):
            c=c+1
    #next
    for q2 in XY_1:
        if(q2[0]!=0.0)and(q2[-1]!=0.0):
            d=d+1

    #moving
    feature_0=XY_0[0:c,:]
    feature_1=XY_1[0:d,:]
    #data have devided into two parts!
    return feature_0,feature_1
#visualization without scaling!
def view_before(filename):
    f=C1.load_tonumpy(filename)
    m,n=shape(f)
    feature_0,feature_1=devide(f)
    #data have devided into two parts!
    #ploting!
    #y=0
    plt.scatter(feature_0[:,0],feature_0[:,1],s=m,c='r',marker='*')
    #y=1
    plt.scatter(feature_1[:,0],feature_1[:,1],s=m,c='b',marker='o')
    #showing()
    plt.show()
#feature scaling!
def Logistic_scaling(f):
    m,n=shape(f)
    #normalization_(0,1)
    for i in range(n-1):
        f[:,i]=(f[:,i]-f[:,i].min())/(f[:,i].max()-f[:,i].min())
    return f
#viewing the scaling data
def Logistic_universe_form(f):
    #the first column need to be all 1
    m,n=shape(f)
    f_universe=zeros([m,n+1])
    f_universe[:,0]=1.0
    f_universe[:,1:]=f
    return f_universe
def view_after_scaling(filename):
    f=C1.load_tonumpy(filename)
    m,n=shape(f)
    #scaling
    f_scaling=Logistic_scaling(f)
    #devide into two parts
    feature_0,feature_1=devide(f_scaling)
    #ploting!
    #y=0
    plt.scatter(feature_0[:,0],feature_0[:,1],s=m,c='r',marker='*')
    #y=1
    plt.scatter(feature_1[:,0],feature_1[:,1],s=m,c='b',marker='o')
    #showing()
    plt.show()










