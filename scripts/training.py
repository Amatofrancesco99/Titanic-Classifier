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
from logistic_regression import *
from other_functions import *

print('\033[92m\033[1m\n\n------------------   1. TRAINING RESULTS  --------------------\033[0m')


# Load the training data
train_X, train_Y = load_file("../titanic-data/titanic-train.txt")


''' Understand the best learning rate value '''
# Which is a good value for the learning rate?
learning_rate_accuracy = {}
learning_rate_values = 1e-4, 0.5e-4, 1e-3, 0.5e-3, 1e-2, 0.05e-1
for lr in learning_rate_values:
    w, b, iteration_loss = logreg_train(train_X, train_Y, lr)
    P = logreg_inference(train_X, w, b)
    Y_hat_train = (P >= 0.5)
    accuracy = (train_Y == Y_hat_train).mean() * 100
    learning_rate_accuracy[lr] = accuracy
# Show how the accuracy goes for specific learning rate values
display_lr_vs_accuracy(learning_rate_accuracy)
# Find the best learning rate value, such that it maximizes the accuracy
best_lr_value, max_accuracy = find_best_lr_value(learning_rate_accuracy)
# What is the training accuracy of the trained model?
print('\nBest learning rate value:', best_lr_value, ' Accuracy:', round(max_accuracy, 2))


''' Once understood the best learning rate, now lets understand which is the best value for the iterations '''
w, b, iteration_loss = logreg_train(train_X, train_Y, best_lr_value)
# Show how the loss goes for specific iterations value
display_iteration_vs_loss(iteration_loss)
# Find the best iterations value, such that it minimizes the loss function
best_iterations_value, min_loss = find_best_iterations_value(iteration_loss)
# How many iterations are required to converge?
print('\nBest iterations value:', best_iterations_value, ' Loss:', round(min_loss, 2))


''' Extra: Load the obtained parameters into an external .csv file '''
# Obtain the parameters considering the "best model", using the previously obtained values
# Why 60000 as iteration value? Time-performance trade-off...by increasing the number of iterations over
# 60000 the loss function doesn't decreases so much and the time spent it's not worth it
w, b, iteration_loss = logreg_train(train_X, train_Y, best_lr_value, 60000)
# Load the parameters into the 'parameter.csv' file
np.savetxt( '../parameters.csv', np.append(w, b))


# END
print('\n')