__author__ = 'John'
import coursera_loaddata_numpy as C1  # my loading programming!
from numpy import *
#plot 2D figure,,not 3D,so only plt is enough!
import matplotlib.pyplot as plt
def view_before(filename):
    f=C1.load_tonumpy(filename)
    m,n=shape(f)
    #classify into two class by lable,contained by numpy.list
    XY_0=[]
    XY_1=[]
    for qq in f:
        if qq[2]==0.0:
            XY_0.append(qq)
        else:
            XY_1.append(qq)

    #loading in the list
    X_0=[]
    Y_0=[]
    X_1=[]
    Y_1=[]


