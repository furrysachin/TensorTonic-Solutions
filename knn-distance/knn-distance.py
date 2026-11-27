
import numpy as np

def knn_distance(X_train, X_test, k):

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Handle 1D inputs
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = len(X_train)
    n_test = len(X_test)

    result = np.full((n_test, k), -1, dtype=int)

    for i in range(n_test):

        distances = np.sqrt(
            np.sum((X_train - X_test[i]) ** 2, axis=1)
        )

        indices = np.argsort(distances, kind='stable')

        count = min(k, n_train)

        result[i, :count] = indices[:count]

    return result
