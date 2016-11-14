__author__ = 'John'
#since the lacking of data ,i cannot confirm the rightness of these codes!
#this is mainly using in the non-linear classification case
#here,mainly shows the use of the LogisticRegression model from the sklearn
#import related .py
from numpy import *  # better: import numpy as np
import coursera_loaddata_numpy as C1
import matplotlib.pyplot as plt
import Logistic_data_processing as C2
#import logistic Regression from the sklearn___linear_model
from sklearn.linear_model import LogisticRegression
#####################################################
#####################################################
def Logistic_sklearn_prepare(filename):
    f_not_scaling=C1.load_tonumpy(filename)
    #need scaling
    f_scaling=C2.Logistic_scaling(f_not_scaling)
    #the first column should be all 1
    f_scaling_universe=C2.Logistic_universe_form(f_scaling)
    return f_scaling_universe

def Logistic_sklearn(filenametraing,filenamepredicting):
    f_traing=Logistic_sklearn_prepare(filenametraing)
    f_testing=Logistic_sklearn_prepare(filenamepredicting)
    classifier=LogisticRegression() #default parameters,also you can change related values of parameters
    #first! training
    #using the .fit() method
    classifier.fit(f_traing[:,0:-1],f_traing[:,-1])
    #ok this model finished traing,then using for prediting!
    predict_lable=classifier.predict(f_testing)
    #Predict class labels for samples in X::::::predict(X)
    return predict_lable
