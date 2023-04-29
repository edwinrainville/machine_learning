from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    error_vec = (np.matmul(X, weight) + bias) - y
    sum_error = np.sum(error_vec)
    bias += -(2 * eta * sum_error)
    weight += -(2 * eta * np.matmul(np.transpose(X), error_vec))

    # Create sparse solution for weights
    reg_step = 2 * eta * _lambda
    negative_weight_inds = weight < -reg_step
    zero_inds = np.logical_and(weight > -reg_step, weight < reg_step) 
    positive_weight_inds = weight > reg_step

    weight[negative_weight_inds] = weight[negative_weight_inds] + reg_step
    weight[positive_weight_inds] = weight[positive_weight_inds] - reg_step
    weight[zero_inds] = 0
    
    return weight, bias


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    sum_square_error = np.sum(((np.matmul(X, weight) + bias) - y)**2)
    reg_weight = _lambda * np.sum(np.abs(weight))
    return sum_square_error + reg_weight

@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = start_weight
    old_b: Optional[np.ndarray] = start_bias
    
    converged = False
    while not converged:
        new_w, new_b = step(X, y, old_w, old_b, _lambda, eta)
        converged = convergence_criterion(new_w, old_w, new_b, old_b, convergence_delta=0.1)
        old_w = np.copy(new_w)
        old_b = np.copy(new_b)

    return new_w, new_b

@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    weight_diff = np.max(np.abs(weight - old_w))
    if weight_diff > convergence_delta:
        converged = False
    else:
        converged = True

    return converged


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    # dimensions of synthetic data
    n = 500
    d = 1000
    k = 100

    # create synthetic dataset
    x_dist_obj = np.random.RandomState(seed=446)
    x = x_dist_obj.chisquare(100, (n,d))
    w = np.zeros(d)
    w[:100] = np.arange(1, 101)/k
    noise_variance = 1
    noise_obj = np.random.RandomState(seed=446)
    noise = noise_obj.normal(0, noise_variance, n)
    y = np.matmul(x, w) + noise

    # normalize data
    x_standard = (x - np.mean(x, axis=0))/np.std(x, axis=0)

    # maximum regularization value
    _lambda_max = np.max(2 * np.abs(np.matmul(np.transpose(x_standard), (y - np.mean(y)))))

    # compute weights for differing regularization
    nonzero_weights = []
    weights = []
    false_discovery_rate = []
    true_positive_rate = []
    regularization_vals = []
    for _lambda in np.flip(np.geomspace(_lambda_max, 0.01, num=100)):
        weight, bias = train(x_standard, y, _lambda)
        nonzero_weights.append(np.count_nonzero(weight))
        false_discovery_rate.append(np.count_nonzero(weight[k+1:])/ np.count_nonzero(weight))
        true_positive_rate.append(np.count_nonzero(weight[:k+1])/k)
        regularization_vals.append(_lambda)

    fig, ax = plt.subplots()
    ax.semilogx(regularization_vals, nonzero_weights)
    ax.set_xlabel('Regularization Value')
    ax.set_ylabel('Number of Non-Zero Weights')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(false_discovery_rate, true_positive_rate, color='k')
    ax.set_xlabel('False Discovery Rate')
    ax.set_ylabel('True Positive Rate')
    plt.show()

if __name__ == "__main__":
    main()
