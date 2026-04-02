import numpy as np
from ml_scratch.tree.decision_tree import DecisionTree

def test_decision_tree():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTree(max_depth=2)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
