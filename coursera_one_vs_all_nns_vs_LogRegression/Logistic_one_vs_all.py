__author__ = 'John'
#import
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import handwrite_digital_processing as C1
#traing and predicting together ,using the sklearn_logisticregression
def Logistic_one_vs_all(Xfilename,yfilename):
    f_origion=C1.load_data_array(Xfilename,yfilename)
    #not need scaling,but first column must be all 1
    #standard first!then deviding!
    f_origion_standard=C1.Logistic_multiclass_standard_form(f_origion)
    #g(w0x0+w1x1+....),,so no matter training data or predict data the first column is 1
    #so,deviding directly!
    f_traing,f_testing=C1.dataset_deviding(f_origion_standard)
    x_traing=f_traing[:,0:-1]
    y_traing=f_traing[:,-1]
    x_testing=f_testing[:,0:-1]
    y_testing=f_testing[:,-1]  #reference answer! actual real results!
    #traing,and predicting!
    classifier=LogisticRegression(solver='newton-cg', max_iter=300,multi_class='ovr')
    #L2,no dual,solver is 'newton-cg',multi_class is using one-vs-rest....
    #training
    classifier.fit(x_traing,y_traing)
    #predicting
    y_predict=classifier.predict(x_testing)
    #outputing
    print("the actual real result is: ")
    print y_testing
    print("the predicting result is: ")
    print y_predict
    #ploting the actual results:
    fig=plt.figure()
    for i in range(10):
        #ploting 10 subplots
        ax=fig.add_subplot(2,5,i+1)
        show_features=x_testing[i,1:]  #attention! the first 1 not need here!
        predicting_lable=y_predict[i]
        titles=str(predicting_lable)
        #reshape(20,20)
        x_digital_matrix=show_features.reshape(20,20)
        ax.set_title(titles)
        plt.imshow(x_digital_matrix,cmap=cm.gray)
    #ending
    plt.show()
