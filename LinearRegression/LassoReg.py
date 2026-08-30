import numpy as np

def sign(val):
    if val < 0:
        return -1
    elif val == 0:
        return 0
    elif val > 0:
        return 1


class LassoRegression:
    def __init__(self, alpha = 0.01):
        self._alpha = alpha

    def fit(self, X, y, learning_rate = 0.01, num_iterations = 1000):
        #Normalizing and appending bias term to X
        X_mean, X_std = X.mean(axis=0), X.std(axis=0) 
        X = (X - X_mean) / X_std

        if len(X.shape) == 1:
            X_new = np.c_[np.ones(X.shape), X]
        else:
            X_new = np.c_[np.ones(X.shape[0]), X]

        y = y.flatten()

        #Storing theta and cost history
        theta_history = np.zeros((num_iterations, X_new.shape[1]))
        cost_history = np.zeros(num_iterations)

        #Initializing a random Theta
        theta = np.random.rand(X_new.shape[1])

        for i in range(num_iterations):
            theta_history[i, :] = theta

            #Calculating MSE
            J = (X_new @ theta) - y
            cost = np.mean(J**2) + (self._alpha * np.linalg.norm(theta, ord = 1))
            cost_history[i] = cost

            vectorize_sign = np.vectorize(sign)
            m = X_new.shape[0]
            gradients = (1 / m) * (X.T @ J) + self._alpha * vectorize_sign(theta)

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
        X_new = np.c_[np.ones(X.shape[0], X)]
        theta = self._theta

        return X_new @ theta

from sklearn.linear_model import Lasso
if __name__ == "__main__":
    X = np.linspace(-10, 10, 400)
    y = 4 + 3*X
    noise = np.random.uniform(0, 5, 400)
    y += noise

    lasso_reg = LassoRegression(alpha = 0.01)
    lasso_reg.fit(X, y)
    print('My Lasso Regression Implementation:')
    print('\tTheta:', lasso_reg._theta)

    lasso_sk = Lasso(alpha = 0.01)
    lasso_sk.fit(X.reshape(-1, 1), y)
    print('Scikit-Learn')
    print('\tIntercept:', lasso_sk.intercept_)
    print('\tCoefficients:', lasso_sk.coef_)
    