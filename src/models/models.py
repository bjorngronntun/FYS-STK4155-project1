import numpy as np
import unittest

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

class ModelTest(unittest.TestCase):
    def test_beta_ols(self):
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        y = np.array([1, 2, 2, 1])
        ol = OLS()
        ol.fit(X, y)
        self.assertAlmostEqual(1.5, ol.beta[0])
        self.assertAlmostEqual(0, ol.beta[1])
    def test_predict_ols(self):
        X = np.array([[1, 0], [1, 1], [1, 1], [1, 2]])
        y = np.array([0, 0, 1, 1])
        ol = OLS()
        ol.fit(X, y)
        predictions = ol.predict(np.array([[1, 3]]))
        self.assertAlmostEqual(1.5, predictions[0])

if __name__ == '__main__':
    unittest.main()
