from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    kernel_poly = np.power((np.outer(x_i, x_j) + 1), d)
    return kernel_poly


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    kernel_rbf = np.exp(-gamma * np.power(np.subtract.outer(x_i, x_j), 2))
    return kernel_rbf    


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    kernel = kernel_function(x, x, kernel_param)
    alpha_hat = np.linalg.solve((np.matmul(np.transpose(kernel), kernel) + _lambda*kernel), np.matmul(np.transpose(y), kernel))
    return alpha_hat


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds

    # Reshape the data to access data in folds
    x_folded = x.reshape((fold_size, num_folds))
    y_folded = y.reshape((fold_size, num_folds))
    
    loss = []
    for n in range(num_folds):
        # test data
        x_train = np.delete(x_folded, n, axis=1).flatten()
        y_train = np.delete(y_folded, n, axis=1).flatten()

        # Validation data
        x_validation = x_folded[:,n].flatten()
        y_validation = y_folded[:,n].flatten()

        # Train the model with given parameters
        alpha_hat = train(x_train, y_train, kernel_function, kernel_param, _lambda)

        # Compute the kernel function values
        kernel_validation = kernel_function(x_train, x_validation, kernel_param)
        y_predicted = np.matmul(alpha_hat, kernel_validation)
        mse = np.mean(np.square(y_predicted - y_validation))
        loss.append(mse)

    avg_loss = np.mean(loss)
    return avg_loss


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    # Train the polynomial kernel model
    gamma = 1/np.median(np.square(np.subtract.outer(x, x)))
    _lambda_range = np.geomspace(10**-5, 10**-1, 50)

    rbf_loss = []
    rbf_loss_optimal = 10**10
    _lambda_optimal = 0
    # Search all parameters with a grid search method
    for _lambda in _lambda_range:
        rbf_loss_new = cross_validation(x, y, kernel_function=rbf_kernel, kernel_param=gamma, _lambda=_lambda, num_folds=num_folds)
        rbf_loss.append(rbf_loss_new)

        if rbf_loss_new < rbf_loss_optimal:
            _lambda_optimal = _lambda
            rbf_loss_optimal = rbf_loss_new

    return (_lambda_optimal, gamma)


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You can use gamma = 1 / median((x_i - x_j)^2) for all unique pairs x_i, x_j in x) for this problem. 
          However, if you would like to search over other possible values of gamma, you are welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    # Train the polynomial kernel model
    d_range = np.arange(5, 25, 1)
    _lambda_range = np.geomspace(10**-5, 10**-1, 50)

    poly_loss = []
    poly_loss_optimal = 10**10
    d_optimal = 0
    _lambda_optimal = 0
    # Search all parameters with a grid search method
    for d in d_range:
        for _lambda in _lambda_range:
            poly_loss_new = cross_validation(x, y, kernel_function=poly_kernel, kernel_param=d, _lambda=_lambda, num_folds=num_folds)
            poly_loss.append(poly_loss_new)

            if poly_loss_new < poly_loss_optimal:
                d_optimal = d
                _lambda_optimal = _lambda
                poly_loss_optimal = poly_loss_new

    return (_lambda_optimal, d_optimal)

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    # Find optimal parameters for poly kernel
    (_lambda_optimal_poly, d_optimal) = poly_param_search(x=x_30, y=y_30, num_folds=x_30.size)
    print(f'For the polynomial kernel, the optimal value of lambda = {_lambda_optimal_poly} and' \
          f'the optimal value for d = {np.round(d_optimal, 2)}.')

    # Find optimal parameters for RBF kernel
    (_lambda_optimal_rbf, gamma_optimal) = rbf_param_search(x=x_30, y=y_30, num_folds=x_30.size)
    print(f'For the RBF kernel, the optimal value of lambda = {_lambda_optimal_rbf} and' \
          f'the optimal value for gamma = {np.round(gamma_optimal, 2)}.')

    # Compute values of the true function
    x_range = np.arange(0,1,0.001)
    f_true_vals = f_true(x_range)

    # Train the Polynomial model with optimal parameters
    alpha_hat_poly = train(x_30, y_30, kernel_function=poly_kernel, kernel_param=d_optimal, _lambda=_lambda_optimal_poly)
    kernel_poly = poly_kernel(x_30, x_range, d_optimal)
    y_predicted_poly = np.matmul(np.transpose(alpha_hat_poly), kernel_poly)
    mse_poly = np.mean(np.square(y_predicted_poly - f_true_vals))

    # Train the RBF model with optimal parameters
    alpha_hat_rbf = train(x_30, y_30, kernel_function=rbf_kernel, kernel_param=gamma_optimal, _lambda=_lambda_optimal_rbf)
    kernel_rbf = rbf_kernel(x_30, x_range, gamma_optimal)
    y_predicted_rbf = np.matmul(np.transpose(alpha_hat_rbf), kernel_rbf)
    mse_rbf = np.mean(np.square(y_predicted_rbf - f_true_vals))

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5,5))
    ax1.plot(x_range, f_true_vals, label='True Function')
    ax1.plot(x_range, y_predicted_poly, label=f'Polynomial Kernel Prediction - mse {np.round(mse_poly, 3)}')
    ax1.scatter(x_30, y_30)
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2.plot(x_range, f_true_vals, label='True Function')
    ax2.plot(x_range, y_predicted_rbf, label=f'RBF Kernel Prediction - mse {np.round(mse_rbf, 3)}')
    ax2.scatter(x_30, y_30)
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.show()

if __name__ == "__main__":
    main()
