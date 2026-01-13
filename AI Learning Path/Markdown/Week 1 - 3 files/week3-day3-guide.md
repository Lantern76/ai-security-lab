# Week 3 Day 3: Gradient Descent

## Learning Objectives
- Understand gradients as direction of steepest increase
- Gradient descent as optimization algorithm
- Learning rate and its effects
- Gradient-based attacks

---

## Concept: What is a Gradient?

### The Core Insight

Gradient = Direction of steepest increase

If you're on a hill:
- Gradient points uphill
- Negative gradient points downhill
- To minimize loss, go opposite of gradient

### Mathematical Definition

For a function f(w):
```
gradient = df/dw = how much f changes when w changes
```

If gradient is positive → increasing w increases f
If gradient is negative → increasing w decreases f

---

## Concept: Gradient Descent

### The Algorithm

```
1. Start with random weights
2. Calculate loss
3. Calculate gradient (which way is uphill?)
4. Step opposite direction (go downhill)
5. Repeat until loss is small
```

### The Update Rule

```python
weights = weights - learning_rate * gradient
```

- `learning_rate`: How big a step to take
- `gradient`: Which direction to go
- Minus sign: Go opposite of gradient (downhill)

---

## Learning Rate Effects

| Learning Rate | Effect |
|---------------|--------|
| Too small | Very slow convergence, may never finish |
| Just right | Smooth descent to minimum |
| Too large | Overshoots, bounces around, may diverge |

---

## Security Thread: Gradient-Based Attacks

### Attack 1: Gradient Manipulation
- Attacker poisons data
- Gradients point wrong direction
- Model descends to wrong minimum

### Attack 2: Adversarial Examples (FGSM)
- Fast Gradient Sign Method
- Use gradient to find direction that maximizes loss
- Small perturbation → wrong prediction
- `x_adversarial = x + epsilon * sign(gradient)`

### Attack 3: Model Extraction
- Query model many times
- Estimate gradients from outputs
- Reconstruct model weights

---

## Syntax Drilling: Gradient Patterns

### Pattern 1: Simple Gradient (1D)
```python
gradient = 2 * (prediction - truth)
```
*Type 5x*

### Pattern 2: Weight Update
```python
weights = weights - learning_rate * gradient
```
*Type 5x*

### Pattern 3: Training Loop Structure
```python
for epoch in range(num_epochs):
    prediction = np.dot(X, weights)
    loss = np.mean((prediction - y) ** 2)
    gradient = calculate_gradient(X, y, weights)
    weights = weights - learning_rate * gradient
```
*Type 5x*

### Pattern 4: Gradient Calculation (Linear Regression)
```python
def calculate_gradient(X, y, weights):
    predictions = np.dot(X, weights)
    error = predictions - y
    gradient = np.dot(X.T, error) / len(y)
    return gradient
```
*Type 5x*

---

## Exercises

### Exercise 1: Manual Gradient Descent
Start with weight = 5, target = 0, learning_rate = 0.1
```
loss = weight²
gradient = 2 * weight
```
Calculate 5 steps of gradient descent by hand.

### Exercise 2: Learning Rate Experiment
What happens with:
- learning_rate = 0.01 (100 steps)
- learning_rate = 0.5 (10 steps)
- learning_rate = 1.5 (diverges?)

### Exercise 3: FGSM Concept
If gradient of loss with respect to input points in direction [0.1, -0.3, 0.2]:
- What direction would you perturb to INCREASE loss?
- How does this fool the model?

---

## Project: Gradient Descent from Scratch

Build a program that:
1. Creates simple data: y = 3*x + noise
2. Starts with random weight
3. Implements gradient descent loop
4. Prints loss each step
5. Shows weight converging to ~3

---

## Rapid Fire Syntax (End of Day)

Type each 3x, no looking:

1. `gradient = 2 * (pred - truth)`
2. `weights = weights - lr * gradient`
3. `for epoch in range(num_epochs):`
4. `loss = np.mean((pred - y) ** 2)`
5. `gradient = np.dot(X.T, error) / len(y)`
6. `predictions = np.dot(X, weights)`

---

## Checklist

- [ ] Can explain gradient as direction of steepest increase
- [ ] Understand the weight update rule
- [ ] Know learning rate effects
- [ ] Understand FGSM attack concept
- [ ] Gradient patterns typed without errors
- [ ] Project completed
