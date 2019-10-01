import numpy as np
import os
import unittest

def variance_beta(X, y_true, y_pred, beta):
    variance_y = np.var(y)
    var_beta = variance_y*(np.linalg.inv(np.dot(X.transpose(), X)))

def mse(y_true, y_pred):
    error = y_true - y_pred
    error_squared = np.dot(error, error)
    observations = error.shape[0]
    return error_squared/observations

def r_squared(y_true, y_pred):
    error = y_true - y_pred
    error_squared = np.dot(error, error)
    mean = np.mean(y_true)
    deviation_from_mean = y_true - mean
    SST = np.dot(deviation_from_mean, deviation_from_mean)
    return 1 - (error_squared/SST)

class EvaluationTest(unittest.TestCase):
    def test_mse_one(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        self.assertEqual(0, mse(y_true, y_pred))
    def test_mse_two(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 2, 2, -2])
        self.assertEqual(9.5, mse(y_true, y_pred))
    def test_r_squared_one(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        self.assertEqual(1, r_squared(y_true, y_pred))
    def test_r_squared_two(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 2])
        self.assertEqual(0.5, r_squared(y_true, y_pred))


if __name__ == '__main__':
    unittest.main()
