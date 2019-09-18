import numpy as np

class K_fold_splitter:
    def __init__(self, m, k):
        self.m = m  # Number of observations
        self.k = k  # Number of folds

    def cross_val_split(self):
        return_values = []
        indices = np.arange(self.m)
        indices2 = indices.copy()
        np.random.seed(5)
        np.random.shuffle(indices2)
        indices_by_fold = np.array_split(indices2, self.k)

        for test_indices in indices_by_fold:

            train_indices = np.delete(indices, test_indices, axis=0)

            return_values.append({
                "train_indices": train_indices,
                "test_indices": test_indices
            })
        return return_values



if __name__ == '__main__':
    kf = K_fold_splitter(50, 7)
    print(kf.cross_val_split())
