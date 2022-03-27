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

''' Elementwise sigmoid function '''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


''' Predict class probabilities '''
def logreg_inference(X, w, b):
    z = X @ w + b
    return sigmoid(z)


''' Get the cross entropy '''
def cross_entropy(P, Y):
    return (-Y * np.log(P) - (1 -Y) * np.log(1 - P)).mean()


''' Train a binary classifier based on L2-regularized logistic regression '''
# lr = learning rate
def logreg_train(X, Y, lr = 1e-3, steps = 100000, lambda_ = 0):
    iteration_loss = {}
    m = X.shape[0] #number of training samples
    n = X.shape[1] #number of features
    w = np.zeros(n)
    b = 0
    for step in range(steps + 1):
        P = logreg_inference(X, w, b)
        if (step % 1000 == 0):
            iteration_loss[step] = cross_entropy(P, Y)
        grad_b = (P - Y).mean()
        grad_w = (X.T @ (P - Y)) / m + 2 * lambda_ * w
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, iteration_loss