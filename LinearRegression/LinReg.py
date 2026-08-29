import numpy as np

class LinearRegression:
    def __init__(self, type = None):
        print("Initialized a Linear Regression Model")
        self._theta = None
        self._coef = None
        self._type = type

    def fit(self, X, y, type = 'normal', learning_rate = 0.1, num_iterations = 1000, batch_size = 10):
        #Normalizing my X data and flattening my y
        X_mean, X_std = X.mean(), X.std()
        X = (X - X_mean) / X_std
        y = y.flatten()

        if type == 'normal':
            #Appending a column of 1s to the X for matrix multiplication
            X_new = np.c_[np.ones(X.shape[0]), X]
            theta = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y

        if type == 'pseudoinverse':
            #Appending a column of 1s to the X for matrix multiplication
            X_new = np.c_[np.ones(X.shape[0]), X]

            theta = np.linalg.pinv(X_new) @ y

        if type == 'batch':
            #Appending a column of 1s to the X for matrix multiplication
            X_new = np.c_[np.ones(X.shape[0]), X]

            #Initializing a random value for theta
            theta = np.random.rand(X_new.shape[1])

            #Creating an array called theta_history to store all the old theta values
            theta_history = np.zeros((num_iterations, X_new.shape[1]))
            cost_history = np.zeros(num_iterations)

            for i in range(num_iterations):
                theta_history[i, :] = theta

                #Calculating MSE
                J = (X_new @ theta) - y
                cost_history[i] = np.mean(J**2)

                #Calculating the Gradient
                m = X_new.shape[0]
                gradients = (1 / m) * (X_new.T @ J)

                theta -= learning_rate * gradients

            theta_history_unscaled = np.zeros_like(theta_history)
            theta_history_unscaled[:, 1:] = theta_history[:, 1:] / X_std
            theta_history_unscaled[:, 0] = theta_history[:, 0] - np.sum(theta_history[:, 1:] * X_mean / X_std, axis = 1)

            self._theta_history = theta_history_unscaled            
            self._cost_history = cost_history
        if type == 'minibatch':
            #Appending a column of 1s to the X for matrix multiplication
            X_new = np.c_[np.ones(X.shape[0]), X]

            #Initializing a random value for theta
            theta = np.random.rand(X_new.shape[1])

            #Creating an array called theta_history to store all the old theta values
            theta_history = np.zeros((num_iterations, X_new.shape[1]))
            cost_history = np.zeros(num_iterations)

            for i in range(num_iterations):
                theta_history[i, :] = theta

                #Selecting the random rows to make a minibatch
                idx = np.random.choice(X_new.shape[0], size = batch_size, replace = False) 
                X_minibatch = X_new[idx]
                y_minibatch = y[idx]

                #Calculating MSE
                J = (X_minibatch @ theta) - y_minibatch
                cost_history[i] = np.mean(J**2)

                #Calculating the Gradient
                gradients = (1 / batch_size) * (X_minibatch.T @ J)

                theta -= learning_rate * gradients

            theta_history_unscaled = np.zeros_like(theta_history)
            theta_history_unscaled[:, 1:] = theta_history[:, 1:] / X_std
            theta_history_unscaled[:, 0] = theta_history[:, 0] - np.sum(theta_history[:, 1:] * X_mean / X_std, axis = 1)

            self._theta_history = theta_history_unscaled            
            self._cost_history = cost_history

        if type == 'stochastic':
            m = X.shape[0]

            #Appending a column of 1s to X for the bias term
            X_new = np.c_[np.ones(X.shape[0]), X]

            #Initializing a random value for theta and a 2D array to store all historical theta values
            theta = np.random.rand((X_new.shape[1]))
            theta_history = np.zeros((num_iterations, X_new.shape[1]))
            cost_history = np.zeros(num_iterations)

            for i in range(num_iterations):
                theta_history[i, :] = theta

                mi = np.random.randint(m)
                xi = X_new[mi, :]
                yi = y[mi]

                J = (xi @ theta) - yi
                cost_history[i] = J**2

                #Calculating the Gradient
                gradient = J * xi

                theta -= learning_rate * gradient


            theta_history_unscaled = np.zeros_like(theta_history)
            theta_history_unscaled[:, 1:] = theta_history[:, 1:] / X_std
            theta_history_unscaled[:, 0] = theta_history[:, 0] - np.sum(theta_history[:, 1:] * X_mean / X_std, axis = 1)

            self._theta_history = theta_history_unscaled            
            self._cost_history = cost_history

        #Unscaling theta values to be stored
        theta_unscaled = np.zeros_like(theta)
        theta_unscaled[1:] = theta[1:] / X_std
        theta_unscaled[0] = theta[0] - np.sum(theta[1:] * X_mean / X_std)

        self._theta = theta_unscaled

    def predict(self, X): 
        X = (X - X.mean()) - X.std()
        X_new = np.c_[np.ones(X.shape[0]), X]
        theta = self._theta

        return X_new @ theta

if __name__ == "__main__":
    X = np.random.rand(100, 1)
    y = 4 + 3 * X * np.random.rand(100, 1)

    lin_reg = LinearRegression()

    lin_reg.fit(X, y)
    print("Normal Equation: ")
    print("\tTheta: ", lin_reg._theta)

    lin_reg.fit(X, y, type = 'pseudoinverse')
    print("Pseudoinverse: ")
    print("\tTheta: ", lin_reg._theta)

    lin_reg.fit(X, y, type = 'batch')
    print("Batch Gradient Descent: ")
    print("\tTheta: ", lin_reg._theta)

    lin_reg.fit(X, y, type = 'stochastic')
    print("Stochastic Gradient Descent: ")
    print("\tTheta: ", lin_reg._theta)

    lin_reg.fit(X, y, type = 'minibatch')
    print("Mini Batch Gradient Descent: ")
    print("\tTheta: ", lin_reg._theta)
