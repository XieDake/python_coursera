__author__ = 'John'
#we need numpy
#coursera_loaddata_numpy
from numpy import *
#defining method loading data.txt
def load_tonumpy(filename):
    #load data to numpy array named f
    f = loadtxt(filename, delimiter='\t', dtype=float)  # store type is string, delimiter can be changed according to the file.
    m, n = shape(f)
    print("the shape of the array is:(%d,%d)" % (m,n))
    print("loadng is working successfully! and please prerare to take our results!")
    return f

