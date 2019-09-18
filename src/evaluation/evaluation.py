import numpy as np
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('.'))))
from src.resampling.resampling import K_fold_splitter

def mse(y_true, y_pred):
    error = y_true - y_pred
    error_squared = np.dot(error, error)
    observations = error.shape[0]
    return error_squared/observations

def r_squared(y_true, y_pred):
    mean = np.mean(y_pred)
    deviation_from_mean = y_true - mean
    SST = np.dot(deviation_from_mean, deviation_from_mean)
    return (1 - mse(y_true, y_pred)/SST)

def cross_val_mse(X, y, k):
    kfs = K_fold_splitter(X.shape[0], k)
    fold_indices = kfs.cross_val_split()
    for fold in fold_indices:
        X_train = X[fold['train_indices']]
        y_train = y[fold['train_indices']]
        X_test = X[fold['test_indices']]
        y_test = y[fold['test_indices']]

if __name__ == '__main__':
    y = np.array([2.0, 5.0, 3.0, 4.0])
    predictions = np.array([2.1, 5.2, 3.05, 3.8])
    mean_squared_error = mse(y, predictions)
    print(mean_squared_error)
