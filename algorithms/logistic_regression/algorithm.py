
import numpy as np

class LogisticRegression:

  def __init__(self, lr=0.001, n_iters=1000):
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None
    
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
      linear_output = np.dot(X, self.weights) + self.bias
      y_predicted = self._sigmoid(linear_output)

      dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
      db = (1/n_samples) * np.sum(y_predicted-y)

      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db
      
    return self
  
  def predict(self, X):
    linear_output = np.dot(X, self.weights) + self.bias
    y_predicted = self._sigmoid(linear_output)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)
  
  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  
  