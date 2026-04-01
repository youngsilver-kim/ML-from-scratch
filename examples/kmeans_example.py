import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from ml_scratch.cluster.kmeans import KMeans

# Generate clustering data
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# Train model
model = KMeans(n_clusters=3, max_iters=100, random_state=42)
model.fit(X)

# Predict cluster labels
labels = model.predict(X)

# Plot clustering result
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker="X", s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
