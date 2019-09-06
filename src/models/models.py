import numpy as np

class OLS():
    def __init__(self):
        pass
    def fit(self, X, y):
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)
    def predict(self, X):
        return np.dot(X, self.beta)
