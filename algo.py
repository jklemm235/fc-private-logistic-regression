import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import math
from scipy.special import softmax, expit

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

class LogisticRegression_DPSGD(object):

    """
    Logistic Regression Classifier with DP SGD
    Parameters
    ----------
    alpha : float, default=0.1
        learning rate

    max_iter : int, default=100
        number of iterations in stochastic gradient descent

    lambda_ : float, default=0 (no penaly)
        Regularization parameter lambda. L2 regularization

    tolerance : float, optional, default=1e-6
        Value indicating the weight change between epochs in which
        gradient descent should terminated.

    DP : bool, default = False
        If False - uses SGD (standart SGD)
        If True  - uses DP_SGD (differentially private SGD)

    L : int, default=1
        lot/batch size for adding the noise to the randomly selected batch with probability L/n, n - number of samples

    C : float, default=1
        gradient norm bound

    sigma: float, default=5
        noise scale
    """

    def __init__(self, alpha=0.1, max_iter=100, lambda_=0.1, tolerance = 1e-6, DP = False, L=1, C=1, sigma=5, theta = None):
        self.alpha          = alpha
        self.max_iter       = max_iter
        self.lambda_        = lambda_
        self.tolerance      = tolerance
        self.DP             = DP
        self.L              = L
        self.C              = C
        self.sigma          = sigma
        self.theta          = theta
        self.cost = 0

    def predict(self, X, y):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        labels : array, shape [n_samples]
            Predicted class label per sample.
        """
        X = np.append(np.ones([X.shape[0],1]), X, axis=1) #add column to the data for bias

        if len(np.unique(y)) == 2: #binary classification
            pred_y =  self.__sigmoid(np.dot(X,self.theta))
        elif len(np.unique(y)) > 2: #multi class classification
            pred_y = self.__softmax(np.dot(X,self.theta), axis=1)
        else:
            raise ValueError(
                        "This solver needs samples of at least 2 classes"
                        " in the data, but the data contains only one"
                        " class: %r")
        return pred_y

    def __softmax(self, z):
        """
        Softmax function with axes=1

        Parameters
        ----------
        z : float
            linear combinations of weights and sample features
            z = w_0*x_0 + w_1*x_1 + ... + w_n*x_n

        Returns
        -------
        Value of softmax function at z

        """

        return softmax(z, axis=1)

    def __sigmoid(self, z):
        """Logistic (sigmoid) function, inverse of logit function

        Parameters:
        ------------
        z : float
            linear combinations of weights and sample features
            z = w_0*x_0 + w_1*x_1 + ... + w_n*x_n

        Returns:
        ---------
        Value of sigmoid function at z

        """

        return 1 / (1 + np.exp(-z))

    def logLiklihood_loss(self, X, y):

        """Regularizd log-liklihood function with L2 regularization

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features+1]
            feature vectors.

        y : list, shape = [n_samples,]
            target values

        Returns
        -----------
        Value of the cost function for given feature vectors and target values:

        """

        reg_term = self.lambda_ / 2 * np.linalg.norm(self.theta) #l2 penatly

        return -1 * np.sum((y * np.log(self.pred_func(np.dot(X,self.theta)))) + ((1 - y) * np.log(1 - self.pred_func(np.dot(X,self.theta))))) + reg_term

    def init_theta(self, X, y):

        """
        Initialize the model and prepare the features and labels for training

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            feature vectors.

        y : list, shape = [n_samples,]
            target values

        Returns
        -----------
        X - feature vector with bias variable: shape = [n_samples, n_features + 1]
        y - trager values (original or one-hot-encoded): shape = [n_samples,]

        """

        X = np.append(np.ones([X.shape[0],1]), X, axis=1) #add column to the data for bias
        if len(np.unique(y)) == 2: #binary classification
            self.theta=np.ones(X.shape[1])
            self.pred_func = self.__sigmoid #sigmoid
        elif len(np.unique(y)) > 2: #multi class classification
            y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1)) #encoode the target values
            self.theta=np.ones((X.shape[1], y.shape[1]))
            self.pred_func = self.__softmax #softmax
        else:
            raise ValueError(
                        "This solver needs samples of at least 2 classes"
                        " in the data, but the data contains only one"
                        " class: %r")

        return X, y

    def SGD(self, X, y):

        """
        Stochastic Gradient Descent, changes self.theta and self.cost

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            feature vectors.

        y : list, shape = [n_samples,]
            target values

        """

        current_iter = 0
        gradient = 1
        self.cost = []
        X, y = self.init_theta(X, y)

        while (current_iter < self.max_iter and np.sqrt(np.sum(gradient ** 2)) > self.tolerance):

            for i in range(X.shape[0]):
                random_sample = random.randint(0, X.shape[0]-1)
                x_sample = X[random_sample]
                y_sample = y[random_sample]
                error = self.pred_func(np.dot(X,self.theta)) - y
                gradient = (X.T.dot(np.array(error))+ self.lambda_ * self.theta) / X.shape[0]
                self.theta = self.theta - self.alpha * gradient

            current_iter += 1
            self.cost.append(self.logLiklihood_loss(X, y))


    def DP_SGD(self, X, y):

        """
        Differentially Private Stochastic Gradient Descent, changes self.theta

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            feature vectors.

        y : list, shape = [n_samples,]
            target values

        """

        current_iter = 0
        noisy_gradient = 1
        self.cost = []
        X, y = self.init_theta(X, y)

        while (current_iter < self.max_iter and np.sqrt(np.sum(noisy_gradient ** 2)) > self.tolerance):

            for epoch in range(int(X.shape[0]/self.L)):
                randomized_samples = random.sample(range(0,X.shape[0]), self.L) #randomly select the lot/batch with probability L/n, n = X.shape[0]

                lots_gradients = []
                for i in randomized_samples:
                    x_sample = X[i]
                    y_sample = y[i]
                    error = self.pred_func(np.dot(X,self.theta)) - y
                    gradient = (X.T.dot(np.array(error))+ self.lambda_ * self.theta) / X.shape[0]
                    # clip the gradient
                    gradient_norm = math.sqrt(np.sum(gradient ** 2))
                    gradient_clip = gradient / max(1, gradient_norm / self.C)
                    lots_gradients.append(gradient_clip)

                # add noise
                noise = np.random.normal(loc=0,scale=self.C*self.sigma,size=self.theta.shape)
                noisy_gradient = (np.sum(lots_gradients, axis=0) + noise) / self.L
                self.theta = self.theta - self.alpha * noisy_gradient

            current_iter += 1
            self.cost.append(self.logLiklihood_loss(X, y))


    def train(self, X, y):

        """
        Training Logistic Regression with SGD or DP_SGD

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            feature vectors.

        y : list, shape = [n_samples,]
            target values

        """

        if self.DP:
            self.DP_SGD(X, y)
        else:
            self.SGD(X, y)


    def evaluate(self, X, y):

        """
        Evaluating the model, printing accuracy and confusion matrix

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            feature vectors.

        y : list, shape = [n_samples,]
            target values

        """

        y_pred =  self.predict(X, y) # calculate predictions
        if len(np.unique(y)) == 2:
            y_pred_target = [1 if y>0.5 else 0 for y in y_pred] # Convert prediction probabilities to classes with 0.5 decision boundary
        elif len(np.unique(y)) > 2:
            y_pred_target = np.argmax(y_pred, axis=1) # Convert prediction probabilities to classes, assigning class corresponding to a maximum probability

        accuracy = accuracy_score(y, y_pred_target,normalize=True)
        conf_mat = confusion_matrix(y, y_pred_target)
        print("The accuracy of the model :", round(accuracy,3)*100,"%")
        print("Confusion Matrix:\n",conf_mat)

