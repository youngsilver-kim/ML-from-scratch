import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 2 + np.random.randn(100)

# 모델 학습
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X, y)

# 예측
y_pred = model.predict(X)

# 시각화
plt.scatter(X, y)
plt.plot(X, y_pred, color="red")
plt.title("Linear Regression Result")
plt.show()

# loss 그래프
plt.plot(model.loss_history)
plt.title("Loss Curve")
plt.show()
