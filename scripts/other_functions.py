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
import matplotlib.pyplot as plt
import warnings

def load_file(filename):
    data = np.loadtxt(filename)
    return data[:, :-1], data[:, -1]


def display_lr_vs_accuracy(learning_rate_accuracy):
    lr_list, accuracy_list = zip(*sorted(learning_rate_accuracy.items()))
    plt.figure()
    plt.plot(lr_list, accuracy_list)
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy\n%')
    plt.title('Learning rate vs Accuracy')
    plt.show()


def find_best_lr_value(learning_rate_accuracy):
    best_lr_value, max_accuracy = -1, -1
    for lr, accuracy in learning_rate_accuracy.items():
        if accuracy == max(learning_rate_accuracy.values()):
            best_lr_value = lr
            max_accuracy = accuracy
            break
    return best_lr_value, max_accuracy


def display_iteration_vs_loss(iteration_loss):
    iteration_list, loss_list = zip(*sorted(iteration_loss.items()))
    plt.figure()
    plt.plot(iteration_list, loss_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Iterations vs Loss')
    plt.show()


def find_best_iterations_value(iteration_loss):
    best_iterations_value, min_loss = -1, -1
    for iterations, loss in iteration_loss.items():
        if (loss == min(iteration_loss.values())) and (iterations != 0):
            best_iterations_value = iterations
            min_loss = loss
            break
    return best_iterations_value, min_loss


def display_scatterplot_two_mi_features(pclass_data, gender_data, y):
    warnings.filterwarnings("ignore")
    plt.figure()
    # Add gaussian noise to the most important features
    pclass_data += np.random.normal(0, 0.1, pclass_data.shape)
    gender_data += np.random.normal(0, 0.1, gender_data.shape)
    plt.scatter(pclass_data, gender_data, s=14, c=y, cmap='brg')
    plt.title('PClass & Gender vs Survived')
    plt.xlabel('PClass')
    plt.ylabel('Gender')
    plt.xticks((1,2,3), ('First', 'Second', 'Third'))
    plt.yticks((0,1), ('Male', 'Female'))
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(['No','','','','','Yes'])
    cbar.set_label('Survived', rotation=270)
    plt.show()
    warnings.simplefilter('always')


def get_color(P):
    if (round((P*100), 2)) >= 50:
        return '\033[92m'
    else:
        return '\033[91m'