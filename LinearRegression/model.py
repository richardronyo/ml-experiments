import numpy as np

class LinearRegression:
    def __init__(self):
        theta = np.array([])
        x = np.array([1])
    
    def fit(self, X, Y):
        p = X.shape[1]
        new_x = np.zeros(p + 1)
        new_x[0] = 1
        
        self.x = new_x
        self.y_true = Y
        self.theta = np.zeros(p + 1) 


        return

    def gradient_descent(self):
        return
