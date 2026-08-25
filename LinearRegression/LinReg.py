import numpy as np

class LinearRegression:
    def __init__(self):
        print("Initialized a Linear Regression Model")
        self._theta = None
        self._coef = None

    def fit(self, X, y, type = 'normal', learning_rate = 0.1, num_iterations = 1000, batch_size = 10):
        y = y.flatten()

        if type == 'normal':
            #Appending a column of 1s to the X for matrix multiplication
            X_new = np.c_[np.ones(X.shape[0]), X]
            theta = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y

            self._theta = theta
        if type == 'pseudoinverse':
            #Appending a column of 1s to the X for matrix multiplication
            X_new = np.c_[np.ones(X.shape[0]), X]

            theta = np.linalg.pinv(X_new) @ y

            self._theta = theta

        if type == 'batch':
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

            self._theta = current_theta
            self._theta_history = thetas
        if type == 'minibatch':
            #Appending a column of 1s to the X for matrix multiplication
            X_new = np.c_[np.ones(X.shape[0]), X]

            #Initializing a random value for theta
            current_theta = np.random.rand(X_new.shape[1])

            #Creating an array called thetas to store all the old theta values
            thetas = np.zeros((num_iterations, X_new.shape[1]))

            for i in range(num_iterations):
                thetas[i, :] = current_theta

                #Selecting the random rows to make a minibatch
                idx = np.random.choice(X_new.shape[0], size = batch_size, replace = False) 
                X_minibatch = X_new[idx]
                y_minibatch = y[idx]

                #Calculating MSE
                J = (X_minibatch @ current_theta) - y_minibatch

                #Calculating the Gradient
                gradients = (1 / batch_size) * (J @ X_minibatch)

                current_theta -= learning_rate * gradients

            self._theta = current_theta
            self._theta_history = thetas

        if type == 'stochastic':
            m = X.shape[0]

            #Appending a column of 1s to X for the bias term
            X_new = np.c_[np.ones(X.shape[0]), X]

            #Initializing a random value for theta and a 2D array to store all historical theta values
            current_theta = np.random.rand((X_new.shape[1]))
            thetas = np.zeros((num_iterations, X_new.shape[1]))

            for i in range(num_iterations):
                thetas[i, :] = current_theta

                mi = np.random.randint(m)
                xi = X_new[mi, :]
                yi = y[mi]

                J = (xi @ current_theta) - yi

                #Calculating the Gradient
                gradient = J * xi

                current_theta -= learning_rate * gradient

            self._theta = current_theta
            self._theta_history = thetas

    def predict(self, X):
        X_new = np.c_[np.ones(X.shape[0]), X]
        theta = np.r_[self._intercept, self._coef]

        return X_new @ theta

if __name__ == "__main__":
    X = np.random.rand(100, 1)
    y = 4 + 3 * X * np.random.rand(100, 1)

    lin_reg = LinearRegression()

    lin_reg.fit(X, y)
    print("Normal Equation: ")
    print("\tThetas: ", lin_reg._theta)

    lin_reg.fit(X, y, type = 'pseudoinverse')
    print("Pseudoinverse: ")
    print("\tThetas: ", lin_reg._theta)

    lin_reg.fit(X, y, type = 'batch')
    print("Batch Gradient Descent: ")
    print("\tThetas: ", lin_reg._theta)

    lin_reg.fit(X, y, type = 'stochastic')
    print("Stochastic Gradient Descent: ")
    print("\tThetas: ", lin_reg._theta)

    lin_reg.fit(X, y, type = 'minibatch')
    print("Mini Batch Gradient Descent: ")
    print("\tThetas: ", lin_reg._theta)
