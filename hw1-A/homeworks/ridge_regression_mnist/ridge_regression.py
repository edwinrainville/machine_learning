import numpy as np
import matplotlib.pyplot as plt

from utils import load_dataset, problem


@problem.tag("hw1-A")
def train(x: np.ndarray, y: np.ndarray, _lambda: float) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), targets (`y`) and regularization parameter (`_lambda`)
    to train a weight matrix $$\\hat{W}$$.


    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        y (np.ndarray): targets represented as `(n, k)` matrix.
            n is number of observations, k is number of classes.
        _lambda (float): parameter for ridge regularization.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: weight matrix of shape `(d, k)`
            which minimizes Regularized Squared Error on `x` and `y` with hyperparameter `_lambda`.
    """
    n, d = x.shape
    x_transpose = np.transpose(x)
    
    # solve the closed form solution for ridge regression
    weight_mapping = np.dot(x_transpose, x) + (_lambda * np.eye(d))
    weight_matrix = np.linalg.solve(weight_mapping, np.dot(x_transpose, y))

    return weight_matrix

@problem.tag("hw1-A")
def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), and weight matrix (`w`) to generate predicated class for each observation in x.

    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        w (np.ndarray): weights represented as `(d, k)` matrix.
            d is number of features, k is number of classes.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: predictions matrix of shape `(n,)` or `(n, 1)`.
    """
    d, k = w.shape
    n, d = x.shape

    # Create one hot encoding matrix
    e_j = np.eye(k)

    prediction_classes = np.empty(n, dtype=int)
    for i in range(n):
        prediction = np.matmul(np.transpose(w), x[i, :])
        prediction_classes[i] = int(np.argmax(np.matmul(e_j, prediction)))
        
    return prediction_classes


@problem.tag("hw1-A")
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encode a vector `y`.
    One hot encoding takes an array of integers and coverts them into binary format.
    Each number i is converted into a vector of zeros (of size num_classes), with exception of i^th element which is 1.

    Args:
        y (np.ndarray): An array of integers [0, num_classes), of shape (n,)
        num_classes (int): Number of classes in y.

    Returns:
        np.ndarray: Array of shape (n, num_classes).
        One-hot representation of y (see below for example).

    Example:
        ```python
        > one_hot([2, 3, 1, 0], 4)
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        ```
    """
    n = y.size
    one_hot_rep = np.zeros((n, num_classes))
    for i in range(n):
        one_hot_rep[i, y[i]] = 1

    return one_hot_rep


def main():

    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    # Convert to one-hot
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)

    _lambda = 1e-4

    w_hat = train(x_train, y_train_one_hot, _lambda)

    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)

    print("Ridge Regression Problem")
    print(
        f"\tTrain Error: {np.average(1 - np.equal(y_train_pred, y_train)) * 100:.6g}%"
    )
    print(f"\tTest Error:  {np.average(1 - np.equal(y_test_pred, y_test)) * 100:.6g}%")

    # Find 10 images that are misclassified and plot
    incorrect_inds = ~np.equal(y_test_pred, y_test)
    incorrect_images = x_test[incorrect_inds, :]
    incorrect_labels = y_test_pred[incorrect_inds]
    incorrect_images_example = incorrect_images[:10, :].reshape((10, 28, 28))
    incorrect_labels_example = incorrect_labels[:10]
    print(incorrect_labels_example)
    fig, ax = plt.subplots(ncols=5, nrows=2)
    n = 0
    for i in range(2):
        for j in range(5):
            ax[i,j].imshow(incorrect_images_example[n,:])
            ax[i,j].set_title(f'Predicted label = {incorrect_labels_example[n]}')
            n +=1
    plt.show()

if __name__ == "__main__":
    main()
