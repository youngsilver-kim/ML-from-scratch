````markdown
# Machine Learning from Scratch (NumPy Implementation)

This repository contains implementations of core machine learning algorithms built entirely from scratch using **NumPy**, without relying on high-level ML libraries such as scikit-learn.

The purpose of this project is to gain a deep understanding of how machine learning algorithms work internally, including optimization, decision boundaries, clustering, and dimensionality reduction.
````
---

## Key Features

- Implemented fundamental ML algorithms from scratch using NumPy
- Unified interface design (`fit`, `predict`)
- Visualization of model behavior and decision boundaries
- Comparison-ready structure for future benchmarking
- Jupyter notebooks with step-by-step explanation and experiments

---

## Implemented Algorithms

| Model | Type | Description |
|------|------|------------|
| Linear Regression | Supervised | Gradient descent-based regression |
| Logistic Regression | Supervised | Binary classification with sigmoid |
| K-Nearest Neighbors | Supervised | Instance-based learning |
| Decision Tree | Supervised | Recursive partitioning using Gini impurity |
| K-Means | Unsupervised | Centroid-based clustering |
| PCA | Unsupervised | Dimensionality reduction via eigen decomposition |

---

## Project Structure

```bash
ML-from-scratch/
├── ml_scratch/          # Core implementations
│   ├── linear_model/
│   ├── neighbors/
│   ├── tree/
│   ├── cluster/
│   └── decomposition/
├── notebooks/           # Experiments & explanations
├── examples/            # Standalone runnable scripts
├── tests/               # Basic validation tests
├── README.md
└── requirements.txt
```
---

## Notebooks (Experiments & Visualization)

Each notebook demonstrates both the theory and behavior of the algorithm.

* [Linear Regression](./notebooks/01_linear_regression_demo.ipynb)
* [Logistic Regression](./notebooks/02_logistic_regression_demo.ipynb)
* [KNN](./notebooks/03_knn_demo.ipynb)
* [Decision Tree](./notebooks/04_decision_tree_demo.ipynb)
* [KMeans](./notebooks/05_kmeans_demo.ipynb)
* [PCA](./notebooks/06_pca_demo.ipynb)

---

## Implementation Highlights

### 1. Gradient Descent (Linear / Logistic Regression)

* Manual implementation of parameter updates
* Loss tracking for convergence analysis
* Numerical stability handling (e.g., sigmoid clipping)

### 2. Decision Tree

* Recursive tree construction
* Gini impurity-based splitting
* Depth control to prevent overfitting

### 3. K-Means

* Iterative centroid update
* Convergence check using tolerance
* Cluster assignment via Euclidean distance

### 4. PCA

* Covariance matrix computation
* Eigenvalue decomposition
* Projection onto principal components

---

## Example Usage

```python
from ml_scratch.linear_model.linear_regression import LinearRegression

model = LinearRegression(lr=0.05, n_iters=1000)
model.fit(X, y)

predictions = model.predict(X)
```
---

## Design Philosophy

This project focuses on:

* Understanding algorithms at a mathematical level
* Translating equations into working code
* Building intuition through visualization
* Avoiding abstraction layers that hide core logic

---

## Future Improvements

* Add Regularization (L1, L2)
* Implement SVM from scratch
* Add Random Forest and ensemble methods
* Benchmark against scikit-learn implementations
* Add performance optimization (vectorization improvements)

---

## Author

* Developed as part of a machine learning study focused on fundamental understanding and implementation.



