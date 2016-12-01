__author__ = 'John'
#In this part, we set regularization parameter  to zero. Because our
#current implementation of linear regression is trying to t a 2-dimensional ,
#regularization will not be incredibly helpful for a  of such low dimension.
#import
from sklearn import linear_model
import data_processing as Q1
import matplotlib.pyplot as plt
class RegularizationLR:
    a=Q1.DataProcessing()
    traingSet=a.standardForm(a.standardForm()[0])
    testSet=a.standardForm(a.standardForm()[1])
    validationSet=a.standardForm(a.standardForm()[2])
    def modelError(self,train_ylable,train_predic):
        my=train_ylable
        mypredict=train_predic
        merror=my-mypredict
        output=0
        for item in merror:
            output+=merror*merror
        #end!
        return output
    #traing viewing!
    def lrTraining(self):
        f=self.traingSet
        x=f[:,0:-1]
        y=f[:,-1]
        LR=linear_model.LinearRegression()
        LR.fit(x,y)
        print("the weight of training is: ")
        print(LR.coef_)
        #ploting
        #scatter ploting!
        plt.scatter(x,y,color='black')
        plt.plot(x,LR.predict(x),color='blue')
        plt.show()
    #ending!
    #learning curve!diagnosing the algorithm underfit or overfit!
    def lrLearningCurve(self):
        traingSet=self.traingSet
        validationSet=self.validationSet
        #size=2_4_6_8_10_12
        trainError=[]
        valError=[]
        size=[2,4,6,8,10,12]
        for num in size:
            trainSubSet=self.a.creatSubdataSet(num,traingSet)
            valSubSet=self.a.creatSubdataSet(num,validationSet)
            LR=linear_model.LinearRegression()
            LR.fit(trainSubSet[:,0:-1],trainSubSet[:,-1])
            print("the traing weight of this sunsets is: ")
            print(LR.coef_)
            trainError.append(self.modelError(trainSubSet[:,-1],LR.predict(trainSubSet[:,0:-1])))
            valError.append(self.modelError(valSubSet[:,-1],LR.predict(valSubSet[:,0:-1])))
        #learning curve ploting!
        plt.plot(size,trainError,color='red',lable='TrainError')
        plt.plot(size,valError,color='blue',lale='ValidationError')
        plt.xlabel("size of traing set")
        plt.ylabel("error")
        plt.legend(loc='upper right')
    #end!
















