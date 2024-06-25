import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

#%matplotlib inline

# load dataset
X_train, y_train = load_data("data/ex2data1.txt")
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))
# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score')
# Set the x-axis label
plt.xlabel('Exam 1 score')
plt.legend(loc="upper right")
plt.show()


# UNQ_C1
# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    ### START CODE HERE ###
    g = 1 / (1 + np.exp(-z))
    ### END SOLUTION ###

    return g

print ("sigmoid(0) = " + str(sigmoid(0)))

print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)