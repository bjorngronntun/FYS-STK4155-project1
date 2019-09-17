import numpy as np

class OLS:
    def __init__(self):
        pass
    def fit(self, X, y):
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)
    def predict(self, X):
        return np.dot(X, self.beta)

class Ridge:
    def __init__(self, lam):
        self.lam = lam
    def fit(self, X, y):
        n = X.shape[1]
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X) + self.lam*np.eye(n)), X.transpose()), y)
    def predict(self, X):
        return np.dot(X, self.beta)
