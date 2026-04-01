import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from ml_scratch.linear_model.logistic_regression import LogisticRegression
from ml_scratch.model_selection import train_test_split
from ml_scratch.metrics import accuracy_score

X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(lr=0.1, n_iters=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

plt.plot(model.loss_history)
plt.title("Logistic Regression Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
