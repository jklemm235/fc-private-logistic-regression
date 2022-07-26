import os # TODO: only necessary with analyse func
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

def sigmoid(z):
    """Logistic (sigmoid) function, inverse of logit function

    Parameters:
    ------------
    z : float
        linear combinations of weights and sample features
        z = w_0*x_0 + w_1*x_1 + ... + w_n*x_n

    Returns:
    ---------
    Value of logistic function at z

    """

    return 1 / (1 + np.exp(-z))

def logLiklihood_loss(x, y, weights, lambda_=0):

    """Regularizd log-liklihood function (cost function to minimized in logistic
    regression classification with L2 regularization)

    Parameters
    -----------
    x : {array-like}, shape = [n_samples, n_features]
        feature vectors.

    weights : 1d-array, shape = [1, n_features]
        Weights of the model (model parameters)

    y : list, shape = [n_samples,], values = 1|0
        target values

    lambda_ : float
        Regularization parameter lambda. L2 regularization

    Returns
    -----------
    Value of regularized log-liklihood function with the given feature values,
    weights, target values, and regularization parameter

    """
    z = np.dot(x, weights)
    reg_term = lambda_ / 2 * np.dot(weights.T, weights) #l2 penatly

    return -1 * np.sum((y * np.log(sigmoid(z))) + ((1 - y) * np.log(1 - sigmoid(z)))) + reg_term



def SGD(X, y, weights, alpha, max_iter, lambda_=0):

    """Stochastic Gradient Descent

    Parameters
    -----------
    x : {array-like}, shape = [n_samples, n_features]
        feature vectors.

    weights : 1d-array, shape = [1, n_features]
        Weights of the model (model parameters)

    y : list, shape = [n_samples,], values = 1|0
        target values

    alpha : float
        learning rate

    max_iter : int
        number of iterations in stochastic gradient descent

    lambda_ : float
        Regularization parameter lambda. L2 regularization

    Returns
    -----------
    Updated model's weights : shape = [1, n_features],
    Cost - loss history : shape = [1, max_iter]  | or shape = [1, current_iter] if current_iter < max_iter

    """

    current_iter = 0
    gradient = 1
    cost = []
    while (current_iter < max_iter and np.sqrt(np.sum(gradient ** 2)) > 0.0000001):

        randomized_samples = random.sample(range(0,X.shape[0]), X.shape[0])
        for i in randomized_samples:
            x_sample = X[i]
            y_sample = y[i]
            gradient = np.array([(sigmoid(np.dot(x_sample,weights)) - y_sample)])*x_sample + lambda_ * weights
            weights = weights - alpha * gradient

        current_iter += 1
        cost.append(logLiklihood_loss(X, y, weights, lambda_))
    return weights, cost


def DP_SGD(X, y, weights, alpha, max_iter, lambda_, L, C=1, sigma=0):

    """Differentially Private Stochastic Gradient Descent

    Parameters
    -----------
    x : {array-like}, shape = [n_samples, n_features]
        feature vectors.

    weights : 1d-array, shape = [1, n_features]
        Weights of the model (model parameters)

    y : list, shape = [n_samples,], values = 1|0
        target values

    n : int
        number of samples

    alpha : float
        learning rate

    max_iter : int
        number of iterations in stochastic gradient descent

    lambda_ : float
        Regularization parameter lambda. L2 regularization

    L : int
        lot/batch size for adding the noise to the randomly selected batch with probability L/n, n - number of samples

    C : float
        gradient norm bound

    sigma: float
        noise scale

    Returns
    -----------
    Updated differentially private model's weights : shape = [1, n_features],
    Cost - loss history: shape = [1, max_iter]  | or shape = [1, current_iter] if current_iter < max_iter

    """
    n, d = X.shape
    current_iter = 0
    gradient = 1
    cost = []
    while (current_iter < max_iter and np.sqrt(np.sum(gradient ** 2)) > 0.0000001):

        for epoch in range(int(X.shape[0]/L)):
            randomized_samples = random.sample(range(0,X.shape[0]), L) #randomly select the lot/batch with probability L/n

            lots_gradients = []
            for i in randomized_samples:
                x_sample = X[i]
                y_sample = y[i]
                gradient = np.array([(sigmoid(np.dot(x_sample,weights)) - y_sample)])*x_sample + lambda_ * weights
                # clip the gradient
                gradient_norm = math.sqrt(np.sum(gradient ** 2))
                gradient_clip = gradient / max(1, gradient_norm / C)
                lots_gradients.append(gradient_clip)

            # add noise
            noise = np.random.normal(loc=0,scale=C*sigma,size=d)
            gradient_noise = (np.sum(lots_gradients, axis=0) + noise) / L
            weights = weights - alpha * gradient_noise
        cost.append(logLiklihood_loss(X, y, weights, lambda_))
        current_iter += 1

    return weights, cost

def analyse_and_plot(X_test, y_test, weights):
    "Calculate Test accuracy"
    X_test1 = np.array(X_test)
    z = np.dot(X_test1,weights) # Multiply theta with X_test
    y_prob = sigmoid(z) # calculate probabilities
    # TODO: y_pred doesnt work yet with multiple classes
    y_pred = [1 if y>0.5 else 0 for y in y_prob] # Convert probabilities to classes with 0.5 decision boundary
    accuracy = accuracy_score(y_test, y_pred,normalize=True)
    conf_mat = confusion_matrix(y_test,y_pred)
    print("The accuracy of the model :", round(accuracy,3)*100,"%")
    print("Confusion Matrix:\n",conf_mat)

    "Plot training loss"
    plt.figure(figsize=(10,8))
    plt.title('Cost Function Slope')
    plt.plot(cost)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Error Values')
    plt.savefig(os.path.join("output", "plot.pdf"))
