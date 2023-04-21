"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = np.empty(degree)  # type: np.ndarray (degree, 1)
        # You can add additional fields
        self.mean_standard = np.empty(degree)
        self.std_standard = np.empty(degree)

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        X_expanded = np.empty((X.size, degree))
        for j in range(degree):
            X_expanded[:,j] = np.squeeze(np.power(X, j+1))

        return X_expanded

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        # number of data points
        n = X.size

        # polynomial expansion of each point
        X_expanded = self.polyfeatures(X, self.degree)

        # standardize the polynomial expanded data
        self.mean_standard = np.mean(X_expanded, axis=0)
        self.std_standard = np.std(X_expanded, axis=0)
        X_expanded_standard = (X_expanded - self.mean_standard)/self.std_standard

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X_expanded_standard]
        _, degree_with_constant = X_.shape

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(degree_with_constant)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        # number of data points
        n = X.size

        # polynomial expansion of each point
        X_expanded = self.polyfeatures(X, self.degree)

        # standardize the polynomial expanded data
        X_expanded_standard = (X_expanded - self.mean_standard)/self.std_standard

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X_expanded_standard]

        # predict
        return X_.dot(self.weight)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    n = a.size
    mse = (1/n) * np.sum(np.power((a - b), 2))
    return mse

@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    # instantiate model
    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    for i in range(n):
        model.fit(Xtrain[0:(i+1)], Ytrain[0:(i+1)])
        errorTrain[i] = mean_squared_error(model.predict(Xtrain[0:(i+1)]), Ytrain[0:(i+1)])
        errorTest[i] = mean_squared_error(model.predict(Xtest[0:(i+1)]), Ytest[0:(i+1)])

    return [errorTrain, errorTest]