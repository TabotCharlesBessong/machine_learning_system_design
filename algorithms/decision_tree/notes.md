# Decision Tree Algorithm

---

## 🌳 **What is a Decision Tree?**

Imagine you have to decide **whether to play outside** based on the weather:

1. **Is it raining?**

   * Yes → Stay inside.
   * No → Go to the next question.
2. **Is it cold?**

   * Yes → Wear a jacket and go outside.
   * No → Go outside and have fun!

This is like a **flowchart** of decisions, where each question splits the possibilities.

🔑 \*\*In machine learning, a decision tree helps a computer decide by:

* Asking a series of questions (features).
* Splitting data based on answers (thresholds).
* Continuing until it makes a final decision (leaf node).\*\*

---

## 🔢 **The Math Behind It**

Let’s get a little more grown-up here but still simple:

1. **Entropy**:

   * Measures **how messy** the data is. Like, "Are most kids playing outside or are some inside?".
   * Formula:

     $$
     H(y) = -\sum p_i \cdot \log(p_i)
     $$

     where $p_i$ is the probability of each label (like 0 or 1).

2. **Information Gain**:

   * Measures **how much better** the split made the data.
   * Formula:

     $$
     IG = H(parent) - \left(\frac{N_L}{N}H(left) + \frac{N_R}{N}H(right)\right)
     $$

     where $N_L$ and $N_R$ are the number of samples in left and right groups.

👉 A split is **good** if it reduces the messiness (entropy) the most!

---

## 🔎 **How Your Code Implements This**

Let’s walk through **each part of the code** like we’re kids making a decision tree out of LEGO blocks:

---

### 1️⃣ `Node` class

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None
```

🧩 **What does this mean?**

* **Node** is a question in the tree:

  * `feature`: which question to ask? (e.g. “Is it raining?”)
  * `threshold`: the answer boundary (e.g. “rain == yes/no?”).
  * `left` and `right`: what to do next (like the yes/no branches).
  * `value`: the final answer (e.g. “play outside”).
* `is_leaf_node()`: checks if this node is a final answer.

---

### 2️⃣ `DecisionTree` class

```python
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
```

* Sets rules:

  * `min_samples_split`: don’t split unless we have enough kids to ask.
  * `max_depth`: how many questions deep we can go.
  * `n_features`: how many questions we can pick from each time.

---

### 3️⃣ `fit()`

```python
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)
```

* `X`: all the features (the questions we can ask).
* `y`: the answers (play outside or not).
* Picks how many questions to ask (`n_features`).
* Starts growing the tree from the top using `_grow_tree`.

---

### 4️⃣ `_grow_tree()`

```python
    def _grow_tree(self, X, y, depth=0):
```

* Checks **how many samples** and **how many different answers**.
* If:

  * max depth is reached ✅
  * or all answers are the same ✅
  * or too few samples to split ✅
  * → stop and make a leaf node (`_most_common_label`).

Else:

* Randomly pick features to try splitting.
* Find the **best split** using `_best_split()`.
* Then split the data and build left and right trees **recursively**.

---

### 5️⃣ `_best_split()`

```python
    def _best_split(self, X, y, feat_idxs):
```

* For each feature picked, try every possible threshold (like “Is it colder than 10°C?”).
* For each split, compute the **information gain** using `_information_gain()`.
* Pick the feature and threshold with the highest information gain.

---

### 6️⃣ `_information_gain()`

```python
    def _information_gain(self, y, X_column, threshold):
```

* Uses `_entropy(y)` to compute the **parent entropy** (how messy things are).
* Splits the data using `_split()`.
* Computes **weighted average entropy** of left and right.
* Subtracts from parent entropy to get **information gain**.

---

### 7️⃣ `_split()`

```python
    def _split(self, X_column, split_thresh):
```

* Splits the data into:

  * left: <= threshold
  * right: > threshold

---

### 8️⃣ `_entropy()`

```python
    def _entropy(self, y):
```

* Counts how many of each label.
* Computes entropy with:

  * $-\sum p \log(p)$
  * where $p$ is the fraction of each label.

---

### 9️⃣ `_most_common_label()`

```python
    def _most_common_label(self, y):
```

* Finds the label that happens the most.
* Returns it as the **final answer** in leaf nodes.

---

### 🔟 `predict()` & `_traverse_tree()`

```python
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
```

* For each sample:

  * Start at the root.
  * At each node, check if it’s a question (feature).
  * If yes:

    * If value <= threshold → left.
    * Else → right.
  * If leaf → return the answer.

---

## 📦 Putting It All Together

1. `fit()` grows the tree by splitting until the data is pure enough or deep enough.
2. At each node:

   * The algorithm asks **"which question splits the data best?"** using information gain.
   * Keeps splitting until it reaches leaf nodes.
3. `predict()` asks each new sample the same questions to find an answer.

---

## 🖼️ **Example**

Let’s say you have:

| Weather | Temperature | Play? |
| ------- | ----------- | ----- |
| Sunny   | 25°C        | Yes   |
| Rainy   | 15°C        | No    |
| Sunny   | 10°C        | Yes   |
| Rainy   | 5°C         | No    |

* The tree might first split on `Weather == Sunny?`

  * If yes → mostly Yes → leaf node Yes.
  * If no → mostly No → leaf node No.

---

## Summary

🔍 Decision trees:

* Ask questions.
* Split data.
* Use math (entropy, info gain) to decide best questions.
* Keep going until they reach final answers.

Your code:

* Defines nodes and the tree.
* Uses entropy to measure impurity.
* Uses information gain to pick splits.
* Recursively builds a tree.
* Predicts new answers by following the questions.

---

If you’d like, I can expand on **any specific part** or help with **tests** or **visuals**. Let me know! 🚀
