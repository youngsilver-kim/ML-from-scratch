import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from ml_scratch.decomposition.pca import PCA

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Apply PCA
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

# Plot projected data
plt.figure(figsize=(8, 5))
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y)
plt.title("PCA Projection of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
