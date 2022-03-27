# INFO
__author__ = "Amato Francesco"
__date__ = "26 Mar 2022"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ =  "Amato Francesco"
__email__ = "francesco.amato01@universitadipavia.it"
__status__ = "Definitive version"
__copyright__ = "© 2022"


# SCRIPT
import numpy as np
from other_functions import load_file
from logistic_regression import *

print('\033[92m\033[1m\n\n-------------------   3. TESTING RESULTS   --------------------\033[0m')


# Load the test data
test_X, test_Y = load_file("../titanic-data/titanic-test.txt")


# What is the test accuracy of the model?
parameters = np.loadtxt('../parameters.csv', delimiter='\n')
w = parameters[:(len(parameters)-1)]
b = parameters[len(parameters)-1]

P = logreg_inference(test_X, w, b)
Y_hat_test = (P >= 0.5)
accuracy = (test_Y == Y_hat_test).mean() * 100
print('\nAccuracy, using testing values:', round(accuracy, 2))


# Is the model overfitting or underfitting the training set?
''' The model is neither overfitting nor underfitting, because the training and testing accuracy is
almost the same, so our model performances are "good" '''


# How can you increase the performance of the model?
''' I can increase the performance of this model by using different machine learning techniques, such
as support vector machines, trees, QDA. Or we can add new samples to the training set, in order to 
have more data.'''


# END
print('\n')