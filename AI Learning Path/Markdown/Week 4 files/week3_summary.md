# Week 3: Machine Learning Theory - Summary

## What We Built

Week 3 established the foundations of supervised learning through hands-on implementation. Every algorithm was built from scratch using only NumPy.

## Daily Progression

### Day 1: The Learning Loop
**Concept:** Weights flow through predict → error → gradient → update → repeat

**Key Code:**
```python
for epoch in range(50):
    predictions = np.dot(X, weights)
    error = predictions - y
    loss = np.mean(error**2)
    gradient = np.dot(X.T, error) / len(y)
    weights = weights - learning_rate * gradient
```

**Insight:** The gradient points toward increasing loss. We go opposite to decrease it.

### Day 2: Watching Learning Happen
**Concept:** Single-variable gradient descent visualization

**Key Learning:** Learning rate controls step size. Too high = overshooting. Too low = slow convergence.

### Day 3: Multi-Feature Regression
**Concept:** Scaling data for stable training

**Key Code:**
```python
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std
```

**Insight:** Features on different scales cause unstable gradients. Standardization fixes this.

### Day 4: Linear Regression Class
**Concept:** Wrapping the training loop in a reusable class

**Key Pattern:**
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
    
    def fit(self, X, y, epochs=1000):
        # Training loop
    
    def predict(self, X):
        return np.dot(X, self.weights)
```

### Day 5: Logistic Regression (Classification)
**Concept:** Sigmoid function converts regression to probability

**Key Code:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Insight:** Sigmoid squashes any value to (0, 1). Threshold at 0.5 gives binary classification.

### Day 6: Evaluation Metrics
**Concepts:** Confusion matrix, precision, recall, F1 score

**Key Formulas:**
- Precision = TP / (TP + FP) — "Of predicted threats, how many were real?"
- Recall = TP / (TP + FN) — "Of real threats, how many did we catch?"
- F1 = 2 × (precision × recall) / (precision + recall)

**Insight:** Precision and recall trade off. Threshold adjustment shifts the balance.

### Day 7: Train/Test Split
**Concept:** Proper evaluation methodology to detect overfitting

**Key Code:**
```python
# Shuffle indices to mix classes
indices = np.arange(n_samples)
np.random.shuffle(indices)
X_shuffled, y_shuffled = X[indices], y[indices]

# Split
split = int(0.8 * n_samples)
X_train, X_test = X_shuffled[:split], X_shuffled[split:]
y_train, y_test = y_shuffled[:split], y_shuffled[split:]

# Scale using ONLY training statistics
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std
```

**Critical Insight:** 
- Train accuracy (96%) vs Test accuracy (80%) reveals overfitting
- Never leak test data statistics into training pipeline
- 67% recall on test data = 33% of threats missed

## Core Mental Model

```
Data → Model → Prediction
              ↓
         Loss Function ← True Labels
              ↓
          Gradient
              ↓
        Weight Update
              ↓
         Better Model
```

## Security Applications Identified

1. **Data Poisoning:** Corrupt training data to shift decision boundary
2. **Evasion Attacks:** Craft inputs that slip under detection threshold
3. **Feature Engineering as Defense:** Add temporal/behavioral features to catch patterns
4. **Preprocessing as Attack Surface:** Manipulate shuffle/split logic to corrupt label pairings

## Key Equations

**Mean Squared Error (MSE):**
```
loss = (1/n) × Σ(prediction - y)²
```

**Gradient for Linear Regression:**
```
gradient = (1/n) × X.T @ (predictions - y)
```

**Sigmoid:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Logistic Regression Forward Pass:**
```
z = X @ weights
probability = sigmoid(z)
prediction = 1 if probability > threshold else 0
```

## Files Produced

| File | Description |
|------|-------------|
| week3_day1.py | Basic gradient descent loop |
| week3_day2.py | Single-variable learning visualization |
| week3_day3.py | Multi-feature with scaling |
| week3_day4.py | LinearRegression class |
| week3_day5.py | LogisticRegression class |
| week3_day6.py | Evaluation metrics |
| week3_day7.py | Complete pipeline with train/test split |

## Operational Lessons

1. **67% recall is unacceptable in high-security contexts** — one in three threats missed
2. **Precision/recall tradeoff depends on operational context** — sensitive systems bias toward recall
3. **Overfitting creates false confidence** — always evaluate on held-out data
4. **More diverse training data reduces overfitting** — harder to memorize patterns

## Bridge to Week 4

Week 3 established single-layer learning. The limitation: linear decision boundaries only.

Week 4 asks: What if we stack multiple layers?

Answer: The same gradient descent loop applies, but gradients must flow backward through all layers. This is backpropagation — the foundation of deep learning.

The math is identical. The bookkeeping gets harder.
