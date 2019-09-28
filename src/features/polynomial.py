import numpy as np
from itertools import combinations_with_replacement
import unittest

class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
        self.names = []
    def fit_transform(self, X, names):
        columns_no = X.shape[1]
        columns = range(columns_no)
        return_X = np.c_[np.ones((X.shape[0], 1)), X]
        self.names = ['1'] + names

        # Make higher-degree features
        for i in range(2, self.degree + 1):
            column_combinations = list(combinations_with_replacement(columns, i))
            name_combinations = list(combinations_with_replacement(names, i))

            for j in column_combinations:
                new_feature = np.prod(X[:, j], axis=1)
                return_X = np.c_[return_X, new_feature]

            for j in name_combinations:
                new_name = ''
                values = set(j)
                for v in values:
                    instances = j.count(v)
                    if instances == 1:
                        new_name += '({})'.format(v)
                    else:
                        new_name += '({}^{})'.format(v, instances)

                self.names.append(new_name)
        return(return_X)

class PolynomialFeaturesTest(unittest.TestCase):
    def test_one(self):
        X = np.array([[2, 3]])
        pf = PolynomialFeatures(2)
        self.assertEqual([[1, 2, 3, 4, 6, 9]], pf.fit_transform(X).tolist())
    def test_two(self):
        X = np.array([[0.5, 2.0], [0.2, 1.1]])
        pf = PolynomialFeatures(2)
        expected = [[1.0, 0.5, 2.0, 0.25, 1.0, 4.0], [1.0, 0.2, 1.1, 0.04, 0.22, 1.21]]
        for i in range(len(expected)):
            for j in range(len(expected[0])):
                self.assertAlmostEqual(expected[i][j], (pf.fit_transform(X)).tolist()[i][j])




if __name__ == '__main__':
    unittest.main()
