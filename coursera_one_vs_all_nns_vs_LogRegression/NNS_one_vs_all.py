__author__ = 'John'
#import
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import handwrite_digital_processing as C1
#traing and predicting together ,using the sklearn_nns-MLP
def Logistic_one_vs_all(Xfilename,yfilename):
    f_origion=C1.load_data_array(Xfilename,yfilename)
    #no needing adding '1',so,using it directly
    #no standarding!
    #deviding first!
    f_traing,f_testing=C1.dataset_deviding(f_origion)
    x_traing=f_traing[:,0:-1]
    y_traing=f_traing[:,-1]
    x_testing=f_testing[:,0:-1]
    y_testing=f_testing[:,-1]  #reference answer! actual real results!
    #traing,and predicting!
    classifier=MLPClassifier(hidden_layer_sizes=(100,50), activation='logistic', solver='adam',
                             alpha=0.0001, batch_size='auto', learning_rate='constant',
                             learning_rate_init=0.001, power_t=0.5, max_iter=200,
                             shuffle=True, random_state=None, tol=0.0001, verbose=False,
                             warm_start=False, momentum=0.9, nesterovs_momentum=True,
                             early_stopping=True, validation_fraction=0.1, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-08)
    #training!
    classifier.fit(x_traing,y_traing)
    #predicting!
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
        show_features=x_testing[i,:]  #no '1'!
        predicting_lable=y_predict[i]
        titles=str(predicting_lable)
        #reshape(20,20)
        x_digital_matrix=show_features.reshape(20,20)
        ax.set_title(titles)
        plt.imshow(x_digital_matrix,cmap=cm.gray)
    #ending
    plt.show()