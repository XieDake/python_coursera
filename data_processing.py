__author__ = 'John'
#need import sth
import numpy as np
import matplotlib.pyplot as plt
import random
#import ending
class DataProcessing:
    filename1=''
    filename2=''
    filename3=''
    filename4=''
    filename5=''
    filename6=''
    #numpy array
    fx=''
    fxtest=''
    fxval=''
    fy=''
    fytest=''
    fyval=''
    #initialize
    def __init__(self,filename1='X.txt',filename2='Xtest.txt',filename3='Xval.txt',
                 filename4='y.txt',filename5='ytest.txt',filename6='yval.txt'):
        self.filename1=filename1
        self.filename2=filename2
        self.filename3=filename3
        self.filename4=filename4
        self.filename5=filename5
        self.filename6=filename6
        self.fx=np.loadtxt(self.filename1, delimiter=',', dtype=float)
        self.fxtest=np.loadtxt(self.filename2, delimiter=',', dtype=float)
        self.fxval=np.loadtxt(self.filename3, delimiter=',', dtype=float)
        self.fy=np.loadtxt(self.filename4, delimiter=',', dtype=float)
        self.fytest=np.loadtxt(self.filename5, delimiter=',', dtype=float)
        self.fyval=np.loadtxt(self.filename6, delimiter=',', dtype=float)
        #end

    #conbine data into standard form
    def standardForm(self,):
        #numpy array
        fx=self.fx
        fy=self.fy
        fxtest=self.fxtest
        fytest=self.fytest
        fxval=self.fxval
        fyval=self.fyval
        m1=np.shape(fx)
        traingSet=np.zeros([m1[0],2])
        traingSet[:,0]=fx[:]
        traingSet[:,-1]=fy
        m2=np.shape(fxtest)
        testSet=np.zeros([m2[0],2])
        testSet[:,0]=fxtest[:]
        testSet[:,-1]=fytest  #ilegal Cheating!
        m3=np.shape(fxval)
        validationSet=np.zeros([m3[0],2])
        validationSet[:,0]=fxval[:]
        validationSet[:,-1]=fyval
        return traingSet,testSet,validationSet
    #data viewing
    def dataViewing(self,f):
        #2D ploting
        x=f[:,0:-1]
        y=f[:,-1]
        #scatter ploting!
        plt.scatter(x,y,color='black')
        plt.show()
    #ending plot!
    #standard form!
    #the first column must be all 1
    def standard(self,f):
        m,n=np.shape(f)
        fNew=np.zeros([m,n+1])
        fNew[:,0:-1]=f
        fNew[:,-1]=1.0
        return fNew
    def creatSubdataSet(self,num,f):
        #num>=2
        m,n=np.shape(f)
        subDataSet=np.zeros([num,n])
        #random selecting!
        for i in range(num):#这就是产生的子集的数目
            index=random.randint(0,m-1)#随机选取
            subDataSet[i,:]=f[index,:]
        return subDataSet









