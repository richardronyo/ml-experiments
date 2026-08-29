import numpy as np

class PolyScaler:
    def __init__(self, degree = 1, include_bias = False):
        self._degree = degree
        self._bias = include_bias

    def fit_transform(self, X):
        X_poly = np.column_stack([X**i for i in range(1, self._degree + 1)])

        if self._bias:
            X_poly = np.c_[np.ones(X.shape[0]), X_poly]

        return X_poly


if __name__ == "__main__":
    X = np.linspace(-10, 10, 20)
    print("Original X: ", X.shape)
    print(X)
    poly_scaler1 = PolyScaler(degree = 2, include_bias = False)
    X_poly1 = poly_scaler1.fit_transform(X)
    print("\nX Scaled to Degree 2, Without Bias: ", X_poly1.shape)
    print(X_poly1)

    poly_scaler2 = PolyScaler(degree = 2, include_bias = True)
    X_poly2 = poly_scaler2.fit_transform(X)
    print("\nX Scaled to Degree 2, With Bias: ", X_poly2.shape)
    print(X_poly2)
    