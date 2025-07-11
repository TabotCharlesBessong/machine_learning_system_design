# Support Vector Machine (SVM) - Mathematics and Implementation

## What is an SVM?

A Support Vector Machine is a supervised learning algorithm used for classification (and sometimes regression). It finds the optimal hyperplane that separates data points of different classes with the **maximum margin**.

---

## The Mathematics

### 1. The Linear SVM Objective

Given data points $(x_i, y_i)$ where $x_i \in \mathbb{R}^n$ and $y_i \in \{-1, 1\}$, the SVM tries to solve:

$$
\min_{w, b} \frac{1}{2} \|w\|^2 + \lambda \sum_{i=1}^m \max(0, 1 - y_i(w^T x_i + b))
$$

- The first term, $\frac{1}{2} \|w\|^2$, encourages a large margin.
- The second term is the **hinge loss**: $\max(0, 1 - y_i(w^T x_i + b))$ penalizes points on the wrong side or within the margin.
- $\lambda$ is a regularization parameter controlling the trade-off between margin size and classification error.

---

### 2. The Hinge Loss

- If $y_i(w^T x_i + b) \geq 1$, the loss is 0 (correct side, outside margin).
- If $y_i(w^T x_i + b) < 1$, the loss increases linearly as the point moves further into the wrong side.

---

### 3. Gradient Descent Updatek# Support Vector Machine (SVM) - Mathematics and Implementation

## What is an SVM?

A Support Vector Machine is a supervised learning algorithm used for classification (and sometimes regression). It finds the optimal hyperplane that separates data points of different classes with the **maximum margin**.

---

## The Mathematics

### 1. The Linear SVM Objective

Given data points $(x_i, y_i)$ where $x_i \in \mathbb{R}^n$ and $y_i \in \{-1, 1\}$, the SVM tries to solve:

$$
\min_{w, b} \frac{1}{2} \|w\|^2 + \lambda \sum_{i=1}^m \max(0, 1 - y_i(w^T x_i + b))
$$

- The first term, $\frac{1}{2} \|w\|^2$, encourages a large margin.
- The second term is the **hinge loss**: $\max(0, 1 - y_i(w^T x_i + b))$ penalizes points on the wrong side or within the margin.
- $\lambda$ is a regularization parameter controlling the trade-off between margin size and classification error.

---

### 2. The Hinge Loss

- If $y_i(w^T x_i + b) \geq 1$, the loss is 0 (correct side, outside margin).
- If $y_i(w^T x_i + b) < 1$, the loss increases linearly as the point moves further into the wrong side.

---

### 3. Gradient Descent Update

For each sample, update the weights and bias as follows:

- If $y_i(w^T x_i + b) \geq 1$:
  - $w \leftarrow w - \text{lr} \cdot (2\lambda w)$
- Else:
  - $w \leftarrow w - \text{lr} \cdot (2\lambda w - y_i x_i)$
  - $b \leftarrow b - \text{lr} \cdot y_i$

---

## Prediction

After training, predict the class by:

$$
\hat{y} = \text{sign}(w^T x + b)
$$

In the code, we map predictions to 1 (positive class) or 0 (negative class).

---

## References

- [Wikipedia: Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
- [CS231n Notes: Linear SVM](https://cs231n.github.io/linear-classify/#svm)

---

## Note

This implementation is for educational purposes and works for linearly separable or nearly separable data. For non-linear SVMs or large datasets, consider using libraries like scikit-learn.

For each sample, update the weights and bias as follows:

- If $y_i(w^T x_i + b) \geq 1$:
  - $w \leftarrow w - \text{lr} \cdot (2\lambda w)$
- Else:
  - $w \leftarrow w - \text{lr} \cdot (2\lambda w - y_i x_i)$
  - $b \leftarrow b - \text{lr} \cdot y_i$

---

## Prediction

After training, predict the class by:

$$
\hat{y} = \text{sign}(w^T x + b)
$$

In the code, we map predictions to 1 (positive class) or 0 (negative class).

---

## References

- [Wikipedia: Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
- [CS231n Notes: Linear SVM](https://cs231n.github.io/linear-classify/#svm)

---

## Note

This implementation is for educational purposes and works for linearly separable or nearly separable data. For non-linear SVMs or large datasets, consider using libraries like scikit-learn.