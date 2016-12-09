__author__ = 'John'
#polynomial of degree 8.
#import
import numpy as np
from sklearn.linear_model import Ridge #lamda>0
from sklearn.linear_model import LinearRegression #lamda=0
import data_processing as Q1
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
#####################################################################
#starting!
#lamda=0
class PolynomialRegression0:
    a = ''
    traingSet = ''
    testSet = ''
    validationSet = ''
    def __init__(self):
        self.a = Q1.DataProcessing()
        #the first column not need to add '1'!
        self.traingSet = self.a.standardForm()[0]
        self.testSet = self.a.standardForm()[1]
        self.validationSet = self.a.standardForm()[2]
    #ending!
    def modelError(self,train_ylable,train_predic):
        my=train_ylable
        mypredict=train_predic
        merror=my-mypredict
        output=0
        for item in merror:
            output+=item*item
        return output
    def PrTrainging0(self):
        f = self.traingSet
        x = f[:, 0:-1]
        y = f[:, -1]
        xx=np.linspace(-60,60,100)
        #Polynomial!
        poly=PolynomialFeatures(degree=8) #polynomial degree is 8! complicated  to overfit!
        X=poly.fit_transform(x)
        XX=poly.fit_transform(xx.reshape(xx.shape[0],1)) #surface!must!
        LR = LinearRegression()
        LR.fit(X, y)
        print("the weight of training PolynomialRegression is: ")
        print(LR.coef_)
        # ploting
        # scatter ploting!
        plt.scatter(x, y, color='black')
        plt.plot(xx, LR.predict(XX), color='cornflowerblue')
        plt.show()
    #Learning curve!
    #PolyomialRregression Lamda=0;
    def PrLearningCurve(self):
        #the first column need not all 1
        traingSet = self.traingSet
        validationSet =self.validationSet
        # size=2_4_6_8_10_12
        trainError = []
        valError = []
        size = [2, 4, 6, 8, 10, 12]
        for num in size:
            trainSubSet = self.a.creatSubdataSet(num, traingSet)
            xtr=trainSubSet[:,0:-1]
            xytr=trainSubSet[:,-1]
            valSubSet = self.a.creatSubdataSet(num, validationSet)
            xval=valSubSet[:,0:-1]
            xyval=valSubSet[:,-1]
            LR = LinearRegression()
            #Polunomial!
            poly = PolynomialFeatures(degree=8)
            Xtr=poly.fit_transform(xtr)
            Xval=poly.fit_transform(xval)
            #training!
            LR.fit(Xtr,xytr)
            print("the traing weight of this sunsets is: ")
            print(LR.coef_)
            trainError.append((self.modelError(xytr, LR.predict(Xtr)))/num)
            valError.append((self.modelError(xyval, LR.predict(Xval)))/num)
        # learning curve ploting!
        plt.plot(size, trainError, color='red',linewidth=2)  # ,lable='TrainError'
        plt.plot(size, valError, color='blue',linewidth=2)  # lable='ValidationError')
        #plt.legend(loc='upper left')
        plt.show()
        # end!
    #############################################################################################
    #Adjusting the regularization parameter :
    # try different parameters to see how regularization can lead to a better model.
    #####FIRST::LAMDA=1############
    #Lamda=1,,PolynomaialRegression :plot:Fit and LearningCurve!
    def PrTraing1(self):


