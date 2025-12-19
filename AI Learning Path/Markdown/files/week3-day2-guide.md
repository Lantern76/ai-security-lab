# Week 3 Day 2: Loss Functions

## Learning Objectives
- Understand what loss functions measure
- Common loss functions (MSE, MAE, Cross-Entropy)
- Why loss choice matters
- How attackers exploit loss functions

---

## Concept: What is a Loss Function?

### The Core Insight

Loss function = How you measure "wrongness"

```
Loss = f(predictions, truth)
```

Low loss = good predictions
High loss = bad predictions

**Learning goal:** Minimize loss

---

## Common Loss Functions

### Mean Squared Error (MSE)
```python
MSE = mean((predictions - truth)²)
```

- Squares errors → big errors penalized more
- Used for: Regression (predicting numbers)
- Security note: Sensitive to outliers

### Mean Absolute Error (MAE)
```python
MAE = mean(|predictions - truth|)
```

- Absolute value → linear penalty
- More robust to outliers
- Security note: Harder to manipulate with single outlier

### Cross-Entropy Loss
```python
CE = -mean(truth * log(predictions))
```

- Used for: Classification (predicting categories)
- Penalizes confident wrong predictions heavily
- Security note: Confidence manipulation attacks

---

## Security Thread: Exploiting Loss Functions

### Attack 1: Outlier Injection (MSE)
- MSE squares errors
- Inject one extreme outlier
- Dominates the loss
- Model optimizes for outlier, ignores normal data

### Attack 2: Label Flipping
- Change labels on small % of data
- Loss now rewards wrong predictions
- Model learns inverted patterns

### Attack 3: Confidence Manipulation
- Cross-entropy penalizes confident mistakes
- Attacker crafts inputs that cause confident wrong predictions
- Model's loss spikes, training destabilizes

---

## Syntax Drilling: NumPy Patterns

### Pattern 1: Import
```python
import numpy as np
```
*Type 5x*

### Pattern 2: Create Array
```python
data = np.array([1, 2, 3, 4, 5])
```
*Type 5x*

### Pattern 3: Mean
```python
mean_value = np.mean(data)
```
*Type 5x*

### Pattern 4: Element-wise Operations
```python
squared = (predictions - truth) ** 2
```
*Type 5x*

### Pattern 5: MSE Implementation
```python
def mse(predictions, truth):
    return np.mean((predictions - truth) ** 2)
```
*Type 5x*

### Pattern 6: MAE Implementation
```python
def mae(predictions, truth):
    return np.mean(np.abs(predictions - truth))
```
*Type 5x*

---

## Exercises

### Exercise 1: Loss Comparison
Given predictions = [2, 4, 6] and truth = [1, 4, 8]:
- Calculate MSE by hand
- Calculate MAE by hand
- Which is higher? Why?

### Exercise 2: Outlier Impact
Given predictions = [2, 4, 6, 100] and truth = [1, 4, 8, 10]:
- Calculate MSE
- Calculate MAE
- Which is more affected by the outlier?

### Exercise 3: Attack Design
Design a data poisoning attack that exploits MSE. What would you inject?

---

## Project: Loss Function Comparison

Build a program that:
1. Creates sample data with one outlier
2. Calculates MSE
3. Calculates MAE
4. Shows how outlier affects each differently
5. Visualizes the difference

---

## Rapid Fire Syntax (End of Day)

Type each 3x, no looking:

1. `import numpy as np`
2. `np.array([1, 2, 3])`
3. `np.mean(data)`
4. `np.abs(x)`
5. `(predictions - truth) ** 2`
6. `def mse(pred, truth):`
7. `return np.mean((pred - truth) ** 2)`

---

## Checklist

- [ ] Can explain what loss functions measure
- [ ] Know difference between MSE and MAE
- [ ] Understand outlier vulnerability in MSE
- [ ] Identified loss-based attack vectors
- [ ] NumPy patterns typed without errors
- [ ] Project completed
