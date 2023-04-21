import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    print(X.size)
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    model1 = PolynomialRegression(degree=d, reg_lambda=0)
    model1.fit(X, y)
    model2 = PolynomialRegression(degree=d, reg_lambda=0.01)
    model2.fit(X, y)
    model3 = PolynomialRegression(degree=d, reg_lambda=0.1)
    model3.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints1 = model1.predict(xpoints)
    ypoints2 = model2.predict(xpoints)
    ypoints3 = model3.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, "rx")
    # plt.title(f"PolyRegression with d = {d}")
    plt.plot(xpoints, ypoints1, "b-", label='$\lambda = 0$')
    plt.plot(xpoints, ypoints2, "y-", label='$\lambda = 0.01$')
    plt.plot(xpoints, ypoints3, "g-", label='$\lambda = 0.1$')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
