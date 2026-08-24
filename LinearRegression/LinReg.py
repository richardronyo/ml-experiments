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

    def fit_gradient_descent(self, X, y, learning_rate = 0.1, num_iterations = 1000):
        #Making sure y is flattened
        y = y.flatten()

        #Appending a column of 1s to the X for matrix multiplication
        X_new = np.c_[np.ones(X.shape[0]), X]

        #Initializing a random value for theta
        current_theta = np.random.rand(X_new.shape[1])

        #Creating an array called thetas to store all the old theta values
        thetas = np.zeros((num_iterations, X_new.shape[1]))

        for i in range(num_iterations):
            thetas[i, :] = current_theta

            #Calculating MSE
            J = (X_new @ current_theta) - y

            #Calculating the Gradient
            m = X_new.shape[0]
            gradients = (1 / m) * (J @ X_new)

            current_theta -= learning_rate * gradients

        self._thetas = thetas
        self._intercept = current_theta[0]
        self._coef = current_theta[1]
    

if __name__ == "__main__":
    X = np.random.rand(100, 1)
    y = 4 + 3 * X * np.random.rand(100, 1)

    lin_reg = LinearRegression()

    lin_reg.fit_normal(X, y)
    print("Normal Equation: ")
    print("\tIntercept: ", lin_reg._intercept)
    print("\tCoefficient: ", lin_reg._coef)

    lin_reg.fit_pseudoinv(X, y)
    print("Pseudoinverse: ")
    print("\tIntercept: ", lin_reg._intercept)
    print("\tCoefficient: ", lin_reg._coef)

    lin_reg.fit_gradient_descent(X, y)
    print("Gradient Descent: ")
    print("\tIntercept: ", lin_reg._intercept)
    print("\tCoefficient: ", lin_reg._coef)