import numpy as np
import os

def variance_beta(X, y_true, y_pred, beta):
    variance_y = np.var(y)
    var_beta = variance_y*(np.linalg.inv(np.dot(X.transpose(), X)))

def mse(y_true, y_pred):
    error = y_true - y_pred
    error_squared = np.dot(error, error)
    observations = error.shape[0]
    return error_squared/observations

def r_squared(y_true, y_pred):
    mean = np.mean(y_pred)
    deviation_from_mean = y_true - mean
    SST = np.dot(deviation_from_mean, deviation_from_mean)
    return 1 - (mse(y_true, y_pred)/SST)



if __name__ == '__main__':
    y = np.array([2.0, 5.0, 3.0, 4.0])
    predictions = np.array([2.1, 5.2, 3.05, 3.8])
    mean_squared_error = mse(y, predictions)
    print(mean_squared_error)
