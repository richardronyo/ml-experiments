import numpy as np

def RSS(X, Y):
    """
    This function approximates Beta0, and Beta1 using the Residual Sum of Squares
    """
    n = X.shape[0] 

    x_mean = X.mean()
    y_mean = Y.mean()

    x_norm = X - x_mean
    y_norm = Y - y_mean

    beta1_numerator = np.sum(x_norm * y_norm)
    beta1_denominator = np.sum(x_norm ** 2)

    beta1 = beta1_numerator / beta1_denominator
    beta0 = y_mean - (beta1 * x_mean)
    
    rss = 0

    for i in range(n):
        rss += (Y[i] - beta0 - (beta1 * X[i])) ** 2

    return beta0, beta1, rss

def RSE(RSS, n):
    return np.sqrt(RSS / (n - 2))

def SE_B0(RSE, n, X):
    x_mean = X.mean()
    x_norm = X - x_mean
    term_3 = (x_mean ** 2) / (np.sum(x_norm ** 2))

    return RSE * np.sqrt((1 / n) * term_3)
    
def SE_B1(RSE, X):
    x_mean = X.mean()
    x_norm = X - x_mean

    return RSE / np.sqrt(np.sum(x_norm ** 2))

class LinearRegression:
    def __init__(self, X_true, Y_true):
        self._x = X_true
        self._y = Y_true


    def fit(self, X, Y):
        return

    def gradient_descent(self):
        return
