import numpy as np
from ml_scratch.linear_model.linear_regression import LinearRegression

def test_linear_regression():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression(lr=0.1, n_iters=1000)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.allclose(preds, y, atol=1.0)
