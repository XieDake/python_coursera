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
class PolynomialRegression:
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
        m=np.shape(merror)
        M=(m[0]+1)*2
        output=0
        for item in merror:
            output+=(item*item)
        return output/M
    def PrTrainging0(self,degree):
        f = self.traingSet
        x = f[:, 0:-1]
        y = f[:, -1]
        xx=np.linspace(-60,60,100)
        #Polynomial!
        poly=PolynomialFeatures(degree=degree) #polynomial degree is 8! complicated  to overfit!
        X=poly.fit_transform(x)
        XX=poly.fit_transform(xx.reshape(xx.shape[0],1)) #surface!must!
        LR = LinearRegression()
        LR.fit(X, y)
        print("the weight of training PolynomialRegression is: ")
        print(LR.coef_)
        print ("error of traing:")
        print (self.modelError(y,LR.predict(X)))
        # ploting
        # scatter ploting!
        plt.scatter(x, y, color='black')
        plt.plot(xx, LR.predict(XX), color='cornflowerblue')
        plt.show()
    #Learning curve!
    #PolyomialRregression Lamda=0;
    def PrLearningCurve0(self,degree):
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
            ytr=trainSubSet[:,-1]
            valSubSet = self.a.creatSubdataSet(num, validationSet)
            xval=valSubSet[:,0:-1]
            yval=valSubSet[:,-1]
            LR = LinearRegression()
            #Polunomial!
            poly = PolynomialFeatures(degree=degree)
            Xtr=poly.fit_transform(xtr)
            Xval=poly.fit_transform(xval)
            #training!
            LR.fit(Xtr,ytr)
            print("the error of traing is : ")
            print(self.modelError(ytr,LR.predict(Xtr)))
            print("the error of validation is : ")
            print(self.modelError(yval,LR.predict(Xval)))
            trainError.append(self.modelError(ytr,LR.predict(Xtr)))
            valError.append(self.modelError(yval,LR.predict(Xval)))
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
    def PrTraing1(self,alpha,degree):
        f = self.traingSet
        x = f[:, 0:-1]
        y = f[:, -1]
        xx = np.linspace(-60, 60, 100)
        # Polynomial!
        poly = PolynomialFeatures(degree=degree)  # polynomial degree is 8! complicated  to overfit!
        X = poly.fit_transform(x)
        XX = poly.fit_transform(xx.reshape(xx.shape[0], 1))  # surface!must!
        rg=Ridge(alpha=alpha)
        rg.fit(X,y)
        print("the weight of training PolynomialRegression is: ")
        print(rg.coef_)
        # ploting
        # scatter ploting!
        plt.scatter(x, y, color='black')
        plt.plot(xx, rg.predict(XX), color='cornflowerblue')
        plt.show()
    ###########################
    #########lamda=changeing!,,,LearningCurve...........#
    def PrLearningCurve1(self,alpha,degree):
        # the first column need not all 1
        traingSet = self.traingSet
        validationSet = self.validationSet
        #size=2_4_6_8_10_12
        trainError = []
        valError = []
        size = [2, 4, 6, 8, 10, 12]
        for num in size:
            trainSubSet = self.a.creatSubdataSet(num, traingSet)
            xtr = trainSubSet[:, 0:-1]
            ytr = trainSubSet[:, -1]
            valSubSet = self.a.creatSubdataSet(num, validationSet)
            xval = valSubSet[:, 0:-1]
            yval = valSubSet[:, -1]
            RG = Ridge(alpha=alpha)
            # Polunomial!
            poly = PolynomialFeatures(degree=degree)
            Xtr = poly.fit_transform(xtr)
            Xval = poly.fit_transform(xval)
            # training!
            RG.fit(Xtr,ytr)
            print("the error of traing is : ")
            print(self.modelError(ytr, RG.predict(Xtr)))
            print("the error of validation is : ")
            print(self.modelError(yval, RG.predict(Xval)))
            trainError.append((self.modelError(ytr, RG.predict(Xtr))))
            valError.append((self.modelError(yval, RG.predict(Xval))))
        # learning curve ploting!
        plt.plot(size, trainError, color='red', linewidth=2)  # ,lable='TrainError'
        plt.plot(size, valError, color='blue', linewidth=2)  # lable='ValidationError')
        # plt.legend(loc='upper left')
        plt.show()
        # end!
    #Selecting lamada using a cross validation set
    def alphaSelecting(self,degree):
        #range of alpha!
        alphas=[0.01, 0.1, 1, 3, 5, 10, 30, 50, 80, 100, 150, 180, 200, 300, 400, 500, 600, 700, 800,900,1000,1500,2000,2500,3000]
        traingSet = self.traingSet
        xtr=traingSet[:,0:-1]
        ytr=traingSet[:,-1]
        validationSet = self.validationSet
        xval=validationSet[:,0:-1]
        yval=validationSet[:,-1]
        trainError=[]
        valError=[]
        for alpha in alphas:
            RG = Ridge(alpha=alpha)
            # Polunomial!
            poly = PolynomialFeatures(degree=degree)
            Xtr = poly.fit_transform(xtr)
            Xval = poly.fit_transform(xval)
            # training!
            RG.fit(Xtr, ytr)
            print("the error of traing is : ")
            print(self.modelError(ytr, RG.predict(Xtr)))
            print("the error of validation is : ")
            print(self.modelError(yval, RG.predict(Xval)))
            trainError.append((self.modelError(ytr, RG.predict(Xtr))))
            valError.append((self.modelError(yval, RG.predict(Xval))))
        #ploting!##
        ##alpha-CV
        plt.plot(alphas, trainError, color='red', linewidth=2)  # ,lable='TrainError'
        plt.plot(alphas, valError, color='blue', linewidth=2)  # lable='ValidationError')
        # plt.legend(loc='upper left')
        plt.show()




