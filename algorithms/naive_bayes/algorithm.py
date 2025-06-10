import numpy as np

class NaiveBayes:
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.classes = np.unique(y)
    n_classes = len(self.classes)
    self.class2idx = {c: idx for idx, c in enumerate(self.classes)}

    self.priors = np.zeros(n_classes)
    self.means = np.zeros((n_classes, n_features))
    self.vars = np.zeros((n_classes, n_features))

    for idx, c in enumerate(self.classes):
      X_c = X[y == c]
      self.priors[idx] = X_c.shape[0] / n_samples
      self.means[idx, :] = X_c.mean(axis=0)
      self.vars[idx, :] = X_c.var(axis=0) + 1e-9  # add small value for stability

    return self

  def predict(self, X):
    y_pred = [self._predict(x) for x in X]
    return np.array(y_pred)

  def _predict(self, x):
    posteriors = []

    for idx, c in enumerate(self.classes):
      prior = np.log(self.priors[idx])
      class_conditional = np.sum(
        -0.5 * np.log(2 * np.pi * self.vars[idx])
        - 0.5 * ((x - self.means[idx]) ** 2) / self.vars[idx]
      )
      posterior = prior + class_conditional
      posteriors.append(posterior)
    return self.classes[np.argmax(posteriors)]

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)