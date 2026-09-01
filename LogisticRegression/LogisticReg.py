import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def make_classification_data(m=500, n_features=2, class_sep=2.0, seed=42):
    rng = np.random.default_rng(seed)
    
    m0 = m // 2
    m1 = m - m0
    
    # Class centers, offset by class_sep along the first feature
    center0 = np.zeros(n_features)
    center1 = np.zeros(n_features)
    center1[0] = class_sep
    
    X0 = rng.standard_normal((m0, n_features)) + center0
    X1 = rng.standard_normal((m1, n_features)) + center1
    
    y0 = np.zeros(m0)
    y1 = np.ones(m1)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    
    # Shuffle so classes aren't in contiguous blocks
    idx = rng.permutation(m)
    return X[idx], y[idx]

class LogisticRegression:
    def __init__(self):
        print("Initiated a Logistic Regression Model")
        self._theta = None

    def fit(self, X, y, learning_rate = 0.01, num_iterations = 1000):
        #Scaling my X values and flattening y
        X_mean, X_std = X.mean(), X.std()
        X_scaled = (X - X_mean) / X_std
        y = y.flatten()

        if len(X_scaled.shape) == 1:
            X_new = np.c_[np.ones(X_scaled.shape), X_scaled]
        else:
            X_new = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

        #Initializing theta with random values
        theta = np.random.rand(X_new.shape[1])

        #Creating values that will store theta and cost history for Gradient Descent Visualizations
        theta_history = np.zeros((num_iterations, X_new.shape[1]))
        cost_history = np.zeros(num_iterations)

        for i in range(num_iterations):
            m = X_new.shape[0]
            cost = (-1 / m) * (y @ np.log(sigmoid(X_new @ theta)) + (1 - y) @ np.log(1 - sigmoid(X_new @ theta))) 

            theta_history[i, :] = theta
            cost_history[i] = cost

            gradients = (1 / m) * (X_new.T @ (sigmoid(X_new @ theta) - y))
            theta -= learning_rate * gradients

        #Unscaling the theta values to be stored
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

from sklearn.linear_model import LogisticRegression as LR_SK
if __name__ == "__main__":
    logreg = LogisticRegression()
    logreg_sk = LR_SK(solver = 'lbfgs')

    X, y = make_classification_data(m=500, n_features=2, class_sep=2.5)

    logreg.fit(X, y)
    logreg_sk.fit(X, y)

    print('Theta values: ', logreg._theta)
    print('Theta values (sklearn): ', np.r_[logreg_sk.intercept_, logreg_sk.coef_.flatten()])
    fig = plt.figure(figsize = (12, 5))
    ax = fig.add_subplot(111, projection = '3d')

    ax.scatter(logreg._theta_history[:, 0], logreg._theta_history[:, 1], logreg._cost_history, color = 'blue', label = 'Gradient Descent Path') 
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Cost')
    plt.show()