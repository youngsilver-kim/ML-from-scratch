import numpy as np
import matplotlib.pyplot as plt

from ml_scratch.linear_model.linear_regression import LinearRegression

np.random.seed(42)

# Generate synthetic regression data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.squeeze() + np.random.randn(100)

# Train model
model = LinearRegression(lr=0.05, n_iters=1000)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Sort for clean line plotting
sorted_idx = np.argsort(X[:, 0])
X_sorted = X[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

# Plot regression result
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data")
plt.plot(X_sorted, y_pred_sorted, label="Prediction")
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Plot loss curve
if hasattr(model, "loss_history"):
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history)
    plt.title("Linear Regression Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.show()
