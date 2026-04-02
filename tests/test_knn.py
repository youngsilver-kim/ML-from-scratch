import numpy as np
from ml_scratch.neighbors.knn import KNN

def test_knn():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    model = KNN(k=3)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
