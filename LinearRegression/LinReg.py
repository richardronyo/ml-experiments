import numpy as np

class LinearRegression:
    def __init__(self):
        print("Initialized a Linear Regression")
        self._intercept = None
        self._coef = None

    def fit_normal(self, X, y):
        #Appending a column of 1s to the X for matrix multiplication
        X_new = np.c_[np.ones(X.shape[0]), X]
        theta = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y

        self._intercept = theta[0]
        self._coef = theta[1]

    def fit_pseudoinv(self, X, y):
        #Appending a column of 1s to the X for matrix multiplication
        X_new = np.c_[np.ones(X.shape[0]), X]

        theta = np.linalg.pinv(X_new) @ y

        self._intercept = theta[0]
        self._coef = theta[1]



if __name__ == "__main__":
    X = np.random.rand(100, 1)
    print(X.shape)
    y = 4 + 3 * X * np.random.rand(100, 1)

    lin_reg = LinearRegression()
    lin_reg.fit_normal(X, y)
    print("Normal Equation")
    print("\tIntercept: ", lin_reg._intercept)
    print("\tCoefficient: ", lin_reg._coef)

    lin_reg.fit_pseudoinv(X, y)
    print("Pseudoinverse")
    print("\tIntercept: ", lin_reg._intercept)
    print("\tCoefficient: ", lin_reg._coef)
