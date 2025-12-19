# Week 3 Day 6: Logistic Regression

## Learning Objectives
- Understand classification vs regression
- Sigmoid function and probability output
- Binary cross-entropy loss
- Classification attacks

---

## Concept: Classification

### Regression vs Classification

| Regression | Classification |
|------------|----------------|
| Predict numbers | Predict categories |
| Output: any value | Output: probability (0-1) |
| Loss: MSE | Loss: Cross-entropy |
| Example: Predict bytes | Example: Threat or not? |

---

## The Sigmoid Function

Converts any number to probability (0-1):

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

| Input z | Output |
|---------|--------|
| -∞ | 0 |
| 0 | 0.5 |
| +∞ | 1 |

### Logistic Regression Formula

```python
z = X @ weights + bias
probability = sigmoid(z)
prediction = 1 if probability > 0.5 else 0
```

---

## Binary Cross-Entropy Loss

```python
def bce_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

**Why not MSE?**
- MSE gives weak gradients for confident wrong predictions
- BCE heavily penalizes confident mistakes
- Better for classification training

---

## Security Thread: Classification Attacks

### Attack 1: Confidence Manipulation
- Find inputs where model is confident but wrong
- Adversarial examples exploit decision boundaries
- Small perturbation → flip prediction

### Attack 2: Threshold Gaming
- If threshold is 0.5, attacker targets 0.49
- Just barely "safe" but actually malicious
- Defense: Multiple thresholds, human review near boundary

### Attack 3: Label Poisoning
- Flip labels on small % of training data
- Model learns wrong decision boundary
- Malicious activity classified as normal

---

## Syntax Drilling: Classification Patterns

### Pattern 1: Sigmoid Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
*Type 5x*

### Pattern 2: Prediction Probability
```python
z = np.dot(X, weights) + bias
probabilities = sigmoid(z)
```
*Type 5x*

### Pattern 3: Binary Prediction
```python
predictions = (probabilities > 0.5).astype(int)
```
*Type 5x*

### Pattern 4: BCE Loss
```python
def bce_loss(y_true, y_pred):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```
*Type 5x*

### Pattern 5: Gradient for Logistic Regression
```python
error = probabilities - y
gradient_w = np.dot(X.T, error) / len(y)
gradient_b = np.mean(error)
```
*Type 5x*

### Pattern 6: Accuracy Calculation
```python
accuracy = np.mean(predictions == y_true)
```
*Type 5x*

---

## Exercises

### Exercise 1: Sigmoid by Hand
Calculate sigmoid for z = 0, z = 2, z = -2

### Exercise 2: Decision Boundary
If weights = [1, -1], bias = 0, where is the decision boundary?
(Hint: Where does z = 0?)

### Exercise 3: Attack Design
You can modify 5% of training labels. How do you maximize damage to a threat detector?

---

## Project: Threat Classifier

Build:
```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        ...
    
    def sigmoid(self, z):
        ...
    
    def fit(self, X, y, epochs=1000):
        # Training loop with BCE
        ...
    
    def predict_proba(self, X):
        # Return probabilities
        ...
    
    def predict(self, X, threshold=0.5):
        # Return binary predictions
        ...
    
    def accuracy(self, X, y):
        ...
```

Train on synthetic threat data:
- Features: bytes, duration, failed_attempts
- Label: threat (1) or normal (0)

---

## Rapid Fire Syntax (End of Day)

Type each 3x, no looking:

1. `def sigmoid(z):`
2. `return 1 / (1 + np.exp(-z))`
3. `probabilities = sigmoid(np.dot(X, weights) + bias)`
4. `predictions = (probabilities > 0.5).astype(int)`
5. `accuracy = np.mean(predictions == y_true)`
6. `error = probabilities - y`
7. `np.clip(y_pred, epsilon, 1 - epsilon)`

---

## Checklist

- [ ] Understand sigmoid function
- [ ] Know why BCE loss is used for classification
- [ ] Can implement logistic regression from scratch
- [ ] Understand classification-specific attacks
- [ ] Classification patterns typed without errors
- [ ] Project completed (threat classifier)
