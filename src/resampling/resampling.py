import numpy as np

class K_fold_splitter:
    def __init__(self, m, k):
        self.m = m  # Number of observations
        self.k = k  # Number of folds

    def cross_val_split(self, seed=5):
        """
        Returns a list of dictionaries, each of the form
        {
            "train_indices": ...,
            "test_indices": ...
        }
        Each dictionary contains all indices ranging from
        0, ..., m - 1, in such a way that there will be no common elements
        in train and test sets. In the course of the entire list,
        all elements will have been included in exactly one test set.
        """
        return_values = []
        indices = np.arange(self.m)
        indices2 = indices.copy()
        np.random.seed(seed)
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
