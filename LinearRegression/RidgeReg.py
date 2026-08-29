import numpy as np

class RidgeRegresion:
    def __init__(self, alpha = 0.1, type = 'stochastic'):
        self._alpha = alpha
        self._type = type
        self._coefs = None
        self._theta = None

    def fit(self, X, y, learning_rate = 0.01, num_iterations = 1000):
        #Normalizing my X and flattening my y
        X_mean, X_std = X.mean(), X.std()
        X = (X - X_mean) / X_std
        y = y.flatten()

        #Appending the bias term to X
        if len(X.shape) == 1:
            X_new = np.c_[np.ones(X.shape), X]
        else:
            X_new = np.c_[np.ones(X.shape[1]), X]

        if self._type == 'normal':
            A = X_new.T @ X
            theta = np.linalg.inv(A + (self._alpha * np.identity(A.shape[0]))) @ X_new.T @ y

        if self._type == 'stochastic':
            theta = np.random.rand(X_new.shape[1])

            #Storing historical Theta and Cost Function values
            theta_history = np.zeros((num_iterations, X_new.shape[1]))
            cost_history = np.zeros(num_iterations)

            for i in range(num_iterations):
                theta_history[i, :] = theta

                #Calculating Cost
                J = (X_new @ theta) - y
                cost_history[i] = np.mean(J**2) + self._alpha * (theta.T @ theta)

                #Calculating the Gradient
                m = X_new.shape[0]
                gradients = (1 / m) * (X_new.T @ J) + (self._alpha * theta)

                theta -= learning_rate * gradients

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
        X_new = np.c_[np.ones(X.shape[0]), X]
        theta = self._theta
        return X_new @ theta

from sklearn.linear_model import Ridge        
if __name__ == "__main__":
    X = np.linspace(-10, 10, 400)
    y = 4 + 3*X

    noise = np.random.normal(0, 5, 400)
    y += noise

    ridge_reg = RidgeRegresion(alpha = 0.01, type = 'normal')
    ridge_reg.fit(X, y)
    print('Normal Equation')
    print('\tTheta: ', ridge_reg._theta)

    ridge_reg = RidgeRegresion(alpha = 0.01, type = 'stochastic')
    ridge_reg.fit(X, y)
    print('Stochastic')
    print('\tTheta: ', ridge_reg._theta)

    ridge_sk = Ridge(alpha = 0.01)
    ridge_sk.fit(X.reshape(-1, 1), y)
    print('Scikit-Learn')
    print('\tIntercept: ', ridge_sk.intercept_)
    print('\tCoefficients: ', ridge_sk.coef_)