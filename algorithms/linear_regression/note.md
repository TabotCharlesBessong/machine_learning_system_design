# Linear Regression

## Intuition

Linear regression is a fundamental algorithm in machine learning and statistics used to model the relationship between a dependent variable (target) and one or more independent variables (features). The intuition is to find the best straight line (in 2D) or hyperplane (in higher dimensions) that predicts the target variable as accurately as possible from the features.

For a single feature, the model is:

y = wx + b

where:
- `y` is the predicted value,
- `x` is the input feature,
- `w` is the weight (slope),
- `b` is the bias (intercept).

The goal is to find the values of `w` and `b` that minimize the difference between the predicted values and the actual values.

---

## Mathematics Behind Linear Regression

### 1. The Model

For multiple features:
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

where:
- `X` is the matrix of input features,
- `w` is the vector of weights.

### 2. Loss Function (Mean Squared Error)

To measure how well the model fits the data, we use the Mean Squared Error (MSE):
MSE = (1/m) * Σ (yᵢ - ŷᵢ)²
where:
- `m` is the number of samples,
- `yᵢ` is the actual value,
- `ŷᵢ` is the predicted value.

### 3. Gradient Descent

Gradient descent is an optimization algorithm used to minimize the loss function by iteratively updating the weights and bias in the direction of the steepest descent.

The update rules are:
w := w - α * ∂MSE/∂w b := b - α * ∂MSE/∂b

---

## From Math to Python Code

### Example 1: Simple Linear Regression (1 Feature)

#### Mathematical Formula:

ŷ = wx + b
#### Python Implementation:
```python
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Initialize parameters
w = 0.0
b = 0.0
lr = 0.01
epochs = 1000
m = len(X)

for epoch in range(epochs):
  y_pred = w * X + b
  dw = (-2/m) * np.sum(X * (y - y_pred))
  db = (-2/m) * np.sum(y - y_pred)
  w -= lr * dw
  b -= lr * db

print(f"Learned weight: {w:.2f}, bias: {b:.2f}")
# Output should be close to w=2, b=1