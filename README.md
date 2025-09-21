# Linear Regression From Scratch

This repository contains a Python implementation of **Linear Regression** from scratch using **NumPy**, without relying on machine learning libraries like scikit-learn. The project demonstrates how to implement gradient descent, handle multiple features, and predict outcomes for linear regression tasks.

---

## Features

- Single and multiple feature support
- Gradient Descent optimization
- Fit and predict methods
- Get learned parameters (weights and bias)

---

## File

- `vishvaj_linear_regression.py` : Contains the `VishvajLinearRegression` class

---

## Usage

```python
import numpy as np
from vishvaj_linear_regression import VishvajLinearRegression

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([3, 5, 7, 9])  # y = 2*x + 1

# Train model
model = VishvajLinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

# Get learned parameters
w, b = model.get_params()
print("Weights:", w)
print("Bias:", b)

# Make predictions
X_test = np.array([[5], [6]])
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
