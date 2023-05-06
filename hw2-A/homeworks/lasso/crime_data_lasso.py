if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_dataset, problem

def mse(X, y, weight):
    """
    computes mean square error between the predicted and actual values.
    """
    prediction = np.matmul(X, weight)
    return np.mean((prediction - y)**2)

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train = pd.read_table("../../data/crime-data/crime-train.txt")
    df_test = pd.read_table("../../data/crime-data/crime-test.txt")

    # normalize data
    X_mean = np.mean(df_train.iloc[:,1:], axis=0)
    X_std = np.std(df_train.iloc[:,1:], axis=0)
    X_train = (df_train.iloc[:,1:] - X_mean)/X_std
    y_train = df_train.iloc[:,0]
    X_test = (df_test.iloc[:,1:] - X_mean)/X_std
    y_test = df_test.iloc[:,0]

    # maximum regularization value
    _lambda_max = np.max(2 * np.abs(np.matmul(np.transpose(X_train), (y_train - np.mean(y_train)))))

    # compute weights for differing regularization
    nonzero_weights = []
    regularization_vals = []

    # initialize specific weights of interest
    weight_agePct12t29 = []
    weight_pctWSocSec = []
    weight_pctUrban = []
    weight_agePct65up = []
    weight_householdsize = []
    
    # Initialize mse for train and test sets
    mse_train = []
    mse_test = []

    # Initialize weight, bias, and lambda values
    weight = np.zeros(X_train.shape[1])
    bias = 0
    _lambda = _lambda_max
    while _lambda > 0.01:
        weight, bias = train(X_train, y_train, _lambda=_lambda, eta=0.00001, convergence_delta=0.1, 
                             start_weight=weight, start_bias=bias)
        
        # Append specific weight values
        weight_agePct12t29.append(weight['agePct12t29'])
        weight_pctWSocSec.append(weight['pctWSocSec'])
        weight_pctUrban.append(weight['pctUrban'])
        weight_agePct65up.append(weight['agePct65up'])
        weight_householdsize.append(weight['householdsize'])

        # Append number of nonzeros
        nonzero_weights.append(np.count_nonzero(weight))
        regularization_vals.append(_lambda)

        # compute mse 
        mse_train.append(mse(X_train, y_train, weight))
        mse_test.append(mse(X_test, y_test, weight))

        # Reduce the size of lambda by factor of two
        _lambda = _lambda/2

    fig, ax = plt.subplots()
    ax.semilogx(regularization_vals, nonzero_weights)
    ax.set_xlabel('Regularization Value')
    ax.set_ylabel('Number of Non-Zero Weights')
    plt.show()

    fig, ax = plt.subplots()
    ax.semilogx(regularization_vals, weight_agePct12t29, label='agePct12t29')
    ax.semilogx(regularization_vals, weight_pctWSocSec, label='pctWSocSec')
    ax.semilogx(regularization_vals, weight_pctUrban, label='pctUrban')
    ax.semilogx(regularization_vals, weight_agePct65up, label='agePct65up')
    ax.semilogx(regularization_vals, weight_householdsize, label='householdsize')
    ax.set_xlabel('Regularization Value')
    ax.set_ylabel('Weight Values')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.semilogx(regularization_vals, mse_train, label='Training Error')
    ax.semilogx(regularization_vals, mse_test, label='Testing Error')
    ax.set_xlabel('Regularization Value')
    ax.set_ylabel('Mean Square Error')
    ax.set_ylim(0,0.15)
    ax.legend()
    plt.show()

    weight_30, _ = train(X_train, y_train, _lambda=30, eta=0.00001, convergence_delta=0.1, 
                             start_weight=weight, start_bias=bias)

    print(f'The largest weight for lambda = 30 is {weight_30.idxmax()} with value of {weight_30.max()}')
    print(f'The most negative weight for lambda = 30 is {weight_30.idxmin()} with value of {weight_30.min()}')
    print(weight_30)

if __name__ == "__main__":
    main()
