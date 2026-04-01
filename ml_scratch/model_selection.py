import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True):
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_count = int(n_samples * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
