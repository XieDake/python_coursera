__author__ = 'John'
import random
#import matlab datatypr .mat,mainly using this scipy.sio
import scipy.io as sio
#pylab is a sub parts of the matplotlib
#from pylab import *
import matplotlib.pyplot as plt
#gray needing! cm
import matplotlib.cm as cm
import numpy as np
#loading .mat data matrix,and changing it into the numpy.array
def load_data_array(Xfilename,yfilename):
     f_x=np.loadtxt(Xfilename, delimiter='\t',dtype=float)
     f_y=np.loadtxt(yfilename, delimiter='\t',dtype=float)
     m,n=np.shape(f_x)
     #combine (x,y) together!
     f_training=np.zeros([m,n+1])
     #(x,y)
     f_training[:,-1]=f_y
     f_training[:,0:-1]=f_x
     return f_training
#showing_before 100=10*10
def showing_before(f_traing):
    f=f_traing
    #random(100)
    #cycle 100 iterations
    fig=plt.figure()
    for i in range(100):
        rand_num=random.randrange(1,5000,3)
        #ploting 100 subplots
        ax=fig.add_subplot(10,10,i+1)
        x_features=f_traing[rand_num,0:-1]
        y_lable=f_traing[rand_num,-1]
        titles=str(y_lable)
        #reshape(20,10)
        x_digital_matrix=x_features.reshape(20,20)
        ax.set_title(titles)
        plt.imshow(x_digital_matrix,cmap=cm.gray)
    #ending
    plt.show()
#Logistic_regression need stand form data input
def Logistic_multiclass_standard_form(f_traing):
    f=f_traing
    m,n=np.shape(f)
    f_Logic=np.zeros([m,n+1])
    f_Logic[:,0]=1.0
    f_Logic[:,1:]=f
    return f_Logic
#creating traing_set and the testing_set
def dataset_deviding(f_origin):
    f=f_origin
    testing_Reference=[100,500,1000,1500,2000,2500,3000,3500,4000,4500]
    m,n=np.shape(f)
    f_testing=np.zeros([10,n])
    f_traing=np.zeros([m-10,n])
    t1=0
    t2=0
    for i in range(5000):
        if i in testing_Reference:
            f_testing[t1,:]=f[i,:]
            t1+=1
        else:
            f_traing[t2,:]=f[i,:]
            t2+=1
    #end deviding!
    return f_traing,f_testing














