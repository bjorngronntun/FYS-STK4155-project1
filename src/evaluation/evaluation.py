import numpy as np
import os
import unittest

def mse(y_true, y_pred):
    """
    Compute mean squared error

    Parameters
    -----------
    y_true: array-like
            Ground truth target values
    y_pred: array-like
            Predicted target values

    Returns
    -----------
    loss:   float
            Sum of squared differences
            between y_true and y_pred,
            measures how close y_true
            is to y_pred
    """
    error = y_true - y_pred
    error_squared = np.dot(error, error)
    observations = error.shape[0]
    return error_squared/observations

def r_squared(y_true, y_pred):
    """
    Compute r squared (coefficient
    of determination)

    Parameters
    -----------
    y_true: array-like
            Ground truth target values
    y_pred: array-like
            Predicted target values

    Returns
    -----------
    z:      float
            The R^2 score or coefficient
            of determination. Best possible
            score is 1
    """
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
