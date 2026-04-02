import numpy as np
from ml_scratch.decomposition.pca import PCA

def test_pca():
    X = np.random.rand(50, 5)

    model = PCA(n_components=2)
    X_transformed = model.fit_transform(X)

    assert X_transformed.shape == (50, 2)
