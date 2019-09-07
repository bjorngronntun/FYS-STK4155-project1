import numpy as np
def mse(y_true, y_pred):
    error = y_true - y_pred
    error_squared = np.dot(error, error)
    observations = error.shape[0]
    print('Number of observations:', observations)
    return error_squared/observations

if __name__ == '__main__':
    y = np.array([2.0, 5.0, 3.0, 4.0])
    predictions = np.array([2.1, 5.2, 3.05, 3.8])
    mean_squared_error = mse(y, predictions)
    print(mean_squared_error)
