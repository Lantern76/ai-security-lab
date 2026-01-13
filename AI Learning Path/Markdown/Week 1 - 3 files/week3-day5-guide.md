# Week 3 Day 5: Linear Regression

## Learning Objectives
- Understand linear regression as a model
- Implement from scratch
- Evaluate model performance
- Model extraction attacks

---

## Concept: Linear Regression

### What It Does

Finds the best line (or hyperplane) through data.

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
y = X @ w + b
```

### "Best" = Minimizes Squared Error

```
Loss = Σ(prediction - truth)²
```

The weights that minimize this loss = the "learned" model.

---

## The Linear Regression Algorithm

```python
# 1. Initialize weights randomly
# 2. Repeat until converged:
#    a. Predict: y_hat = X @ w
#    b. Calculate loss: MSE
#    c. Calculate gradient
#    d. Update weights
# 3. Return final weights
```

---

## Closed-Form Solution

Linear regression has an exact solution (no iteration needed):

```python
w = np.linalg.inv(X.T @ X) @ X.T @ y
```

This is called the "Normal Equation."

**But:** Gradient descent is more general (works for any differentiable loss).

---

## Security Thread: Model Extraction

### The Attack

Attacker doesn't have your model weights, but can query it.

```
1. Send inputs X₁, X₂, ... Xₙ
2. Receive outputs y₁, y₂, ... yₙ
3. Solve for weights: w = solve(X, y)
```

For linear regression: Just need n queries (n = number of features) to extract exact weights.

### Defense Strategies

1. **Rate limiting:** Limit queries per user
2. **Output perturbation:** Add noise to predictions
3. **Prediction rounding:** Reduce precision
4. **Watermarking:** Embed detectable patterns in model behavior

---

## Syntax Drilling: Linear Regression Patterns

### Pattern 1: Initialize with Bias
```python
n_features = X.shape[1]
weights = np.random.randn(n_features)
bias = 0.0
```
*Type 5x*

### Pattern 2: Prediction with Bias
```python
predictions = np.dot(X, weights) + bias
```
*Type 5x*

### Pattern 3: Gradient for Weights
```python
gradient_w = np.dot(X.T, error) / len(y)
```
*Type 5x*

### Pattern 4: Gradient for Bias
```python
gradient_b = np.mean(error)
```
*Type 5x*

### Pattern 5: Update Both
```python
weights = weights - learning_rate * gradient_w
bias = bias - learning_rate * gradient_b
```
*Type 5x*

### Pattern 6: R² Score (Model Evaluation)
```python
ss_res = np.sum((y - predictions) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)
```
*Type 5x*

---

## Exercises

### Exercise 1: Predict by Hand
Given weights = [2, 3], bias = 1, input = [4, 5]:
Calculate prediction.

### Exercise 2: Extraction Math
You have a model with 3 features (unknown weights).
How many queries do you need to extract exact weights?

### Exercise 3: Defense Analysis
If you add noise N(0, 0.1) to outputs, how does this affect:
- Legitimate users?
- Extraction attackers?

---

## Project: Linear Regression from Scratch

Build a class:
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        ...
    
    def fit(self, X, y, epochs=1000):
        # Training loop
        ...
    
    def predict(self, X):
        ...
    
    def score(self, X, y):
        # Return R²
        ...
```

Test on synthetic data. Verify weights are learned correctly.

---

## Rapid Fire Syntax (End of Day)

Type each 3x, no looking:

1. `weights = np.random.randn(n_features)`
2. `predictions = np.dot(X, weights) + bias`
3. `gradient_w = np.dot(X.T, error) / len(y)`
4. `gradient_b = np.mean(error)`
5. `r2 = 1 - (ss_res / ss_tot)`
6. `def fit(self, X, y, epochs=1000):`
7. `def predict(self, X):`

---

## Checklist

- [ ] Understand linear regression mathematically
- [ ] Can implement training loop with bias
- [ ] Know how to calculate R² score
- [ ] Understand model extraction attack
- [ ] Linear regression patterns typed without errors
- [ ] Project completed (class implementation)
