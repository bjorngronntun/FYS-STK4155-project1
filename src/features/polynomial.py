import numpy as np
from itertools import combinations_with_replacement
class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
    def fit_transform(self, X):
        columns_no = X.shape[1]
        columns = range(columns_no)
        return_X = np.c_[np.ones((X.shape[0], 1)), X]

        # Make higher-degree features
        for i in range(2, self.degree + 1):
            column_combinations = list(combinations_with_replacement(columns, i))

            for j in column_combinations:
                new_feature = np.prod(X[:, j], axis=1)
                return_X = np.c_[return_X, new_feature]

        return(return_X)

if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pf = PolynomialFeatures(2)
    print(pf.fit_transform(X))
