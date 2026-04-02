import numpy as np
from ml_scratch.linear_model.logistic_regression import LogisticRegression

def test_logistic_regression():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
