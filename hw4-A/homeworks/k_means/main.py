if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    trained_centers, errors = lloyd_algorithm(data=x_train, num_centers=10, epsilon=10e-3)

    # Plot the errors
    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Error')
    plt.show()

    # Reshape the center images and plot the images
    trained_centers_img = trained_centers.reshape((10,28,28))
    fig, axs = plt.subplots(nrows=2, ncols=5)
    ind = 0
    for i in range(2):
        for n in range(5):
            axs[i,n].imshow(trained_centers_img[ind,:,:])
            ind +=1
    plt.show()


if __name__ == "__main__":
    main()
