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
 
print('\033[92m\033[1m\n\n------------------   2. ANALYSIS RESULTS  --------------------\033[0m')


# Load the obtained parameters, such that the loss is minimized and the accuracy is maximized
parameters = np.loadtxt('../parameters.csv', delimiter='\n')
w = parameters[:(len(parameters)-1)]
b = parameters[len(parameters)-1]


# What would be your probability to survive?
my_info = (1, 0, 22, 0, 0, 50)
P = logreg_inference(my_info, w, b)
print('\nI would have survived with the', get_color(P), round((P*100), 2), '%\033[0m of probability')


# Looking at the learned weights, how the individual features influence the probability of surviving?
# What kind of passengers was most likely to survive? And what kind to to die?
print('\nObtained weights, for each feature:\n', str(w))


# Draw a scatter plot showing the distribution of the two classes in the plane defined by the two most
# influential features. Comment the plot
# Import the useful data
train_X, train_Y = load_file("../titanic-data/titanic-train.txt")
# As we already discussed, the most important parameter are the one with the higher values for the weights
# In this case the most important features are mainly two: PClass & Gender/Sex
display_scatterplot_two_mi_features(train_X[:, 0], train_X[:, 1], train_Y)

# Understand the probability of surviving considering ONLY gender (male/female) and class (1/2/3)
print('\n\nPlease note that the following results have been generated without considering the effects of the other features (their weights have been rejected)\n')
w = w[:2]
for i in range(0, 2):
    for j in range(1, 4):
        my_info = (j, i)
        if (i == 0):
            gender = "male"
        else:
            gender = "female"
        P = logreg_inference(my_info, w, b)
        print("A", gender, "with a", j, "class ticket, survived with the", get_color(P), round((P*100), 2), "%\033[0m of probability")


# END
print('\n')