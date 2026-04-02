import numpy as np
from ml_scratch.cluster.kmeans import KMeans

def test_kmeans():
    X = np.random.rand(50, 2)

    model = KMeans(n_clusters=3)
    model.fit(X)

    preds = model.predict(X)

    assert preds.shape[0] == X.shape[0]
