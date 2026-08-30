import numpy as np

def sign(val):
    if val < 0:
        return -1
    elif val == 0:
        return 0
    elif val > 0:
        return 1

class ElasticRegression:
    def __init__(self, alpha = 0.0, l1_ratio = 0.5):
        self._alpha = alpha
        self._l1_ratio = l1_ratio

    def fit(self, X, y, learning_rate = 0.01, num_iterations = 1000):
        #Scaling X and appending bias term
        X_mean, X_std = X.mean(axis=0), X.std(axis=0) 
        X_scaled = (X - X_mean) / X_std

        if len(X_scaled.shape) == 1:
            X_new = np.c_[np.ones(X_scaled.shape), X_scaled] 
        else:
            X_new = np.c_[np.ones(X_scaled.shape[1]), X_scaled] 

        y = y.flatten()

        #Initializing theta
        theta = np.random.rand(X_new.shape[1])

        #Creating values that will store theta and cost history for Gradient Descent Visualizations
        theta_history = np.zeros((num_iterations, X_new.shape[1]))
        cost_history = np.zeros(num_iterations)

        for i in range(num_iterations):
            theta_history[i, :] = theta

            J = (X_new @ theta) - y
            cost = np.mean(J**2) + (self._l1_ratio * learning_rate * np.linalg.norm(theta, ord = 1)) + (((1 - self._l1_ratio) / 2) * learning_rate * (theta.T @ theta))
            cost_history[i] = cost

            vectorized_sign = np.vectorize(sign)
            lasso_v = vectorized_sign(theta)
            ridge_v = 2*theta

            m = X_new.shape[0]
            gradients = (1 / m) *(X.T @ J) + (self._l1_ratio * lasso_v) + (((1 - self._l1_ratio) / 2) * ridge_v)

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


from sklearn.linear_model import ElasticNet
if __name__ == "__main__":
    X = np.linspace(-10, 10, 400)
    y = 4 + 3*X
    noise = np.random.normal(0, 5, 400)
    y += noise

    elastic_sk = ElasticNet(alpha = 0.01, l1_ratio = 0.5)
    elastic_sk.fit(X.reshape(-1, 1), y)
    print('Scikit-Learn Elastic:')
    print('\tIntercept:', elastic_sk.intercept_)
    print('\tCoefficient:', elastic_sk.coef_)

    elastic_reg = ElasticRegression(alpha = 0.01, l1_ratio = 0.5)
    elastic_reg.fit(X, y)
    print('My Elastic Regression Implementation: ')
    print('\tTheta', elastic_reg._theta)