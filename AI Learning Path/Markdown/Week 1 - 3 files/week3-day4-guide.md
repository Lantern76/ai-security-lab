# Week 3 Day 4: Training Loop from Scratch

## Learning Objectives
- Build complete training loop
- Understand each component's role
- Implement forward pass, loss, backward pass, update
- Training-time attack awareness

---

## The Complete Training Loop

```python
# Initialize
weights = random
learning_rate = 0.01

# Train
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X, weights)
    
    # Calculate loss
    loss = loss_function(predictions, y)
    
    # Backward pass (gradient)
    gradient = calculate_gradient(X, y, weights)
    
    # Update weights
    weights = weights - learning_rate * gradient
    
    # Monitor
    print(f"Epoch {epoch}, Loss: {loss}")
```

---

## Each Component Explained

### Forward Pass
```
Input → Weights → Prediction
```
Model makes a guess based on current weights.

### Loss Calculation
```
Prediction vs Truth → Single number (how wrong)
```
Measures quality of current weights.

### Backward Pass
```
Loss → Gradient (which direction to adjust)
```
Calculates how to change weights to reduce loss.

### Update
```
Old weights - (learning_rate × gradient) → New weights
```
Takes a step toward better weights.

---

## Security Thread: Training-Time Attacks

### Attack Surface: The Training Loop

| Component | Attack |
|-----------|--------|
| Forward pass | Backdoor injection (trigger patterns) |
| Loss calculation | Loss manipulation |
| Gradient calculation | Gradient poisoning |
| Update step | Learning rate manipulation |
| Data batches | Data poisoning |

### Attack: Backdoor Injection
- Attacker adds images with small trigger (e.g., pixel pattern)
- Labels them as target class
- Model learns: trigger → target class
- At test time: Any image + trigger → misclassified

### Defense: Gradient Monitoring
- Track gradient magnitudes during training
- Unusual spikes may indicate poisoned batches
- Anomaly detection on gradients themselves

---

## Syntax Drilling: Training Loop Patterns

### Pattern 1: Initialization
```python
weights = np.random.randn(n_features)
learning_rate = 0.01
num_epochs = 100
```
*Type 5x*

### Pattern 2: Forward Pass
```python
predictions = np.dot(X, weights)
```
*Type 5x*

### Pattern 3: Loss Calculation
```python
loss = np.mean((predictions - y) ** 2)
```
*Type 5x*

### Pattern 4: Gradient Calculation
```python
error = predictions - y
gradient = np.dot(X.T, error) / len(y)
```
*Type 5x*

### Pattern 5: Weight Update
```python
weights = weights - learning_rate * gradient
```
*Type 5x*

### Pattern 6: Complete Loop
```python
for epoch in range(num_epochs):
    predictions = np.dot(X, weights)
    loss = np.mean((predictions - y) ** 2)
    error = predictions - y
    gradient = np.dot(X.T, error) / len(y)
    weights = weights - learning_rate * gradient
```
*Type 5x*

---

## Exercises

### Exercise 1: Component Identification
Given this code, label each line (forward/loss/gradient/update):
```python
pred = np.dot(X, w)
loss = np.mean((pred - y) ** 2)
grad = np.dot(X.T, pred - y) / len(y)
w = w - 0.01 * grad
```

### Exercise 2: Attack Design
You can modify 1% of training data. How do you inject a backdoor?

### Exercise 3: Defense Design
How would you detect if gradients are being poisoned?

---

## Project: Complete Training from Scratch

Build:
1. Generate synthetic data: y = 2*x1 + 3*x2 + noise
2. Initialize random weights
3. Implement complete training loop
4. Track loss over epochs
5. Print final weights (should be close to [2, 3])
6. Add a simple "gradient spike detector"

---

## Rapid Fire Syntax (End of Day)

Type each 3x, no looking:

1. `weights = np.random.randn(n_features)`
2. `predictions = np.dot(X, weights)`
3. `loss = np.mean((predictions - y) ** 2)`
4. `error = predictions - y`
5. `gradient = np.dot(X.T, error) / len(y)`
6. `weights = weights - learning_rate * gradient`
7. `for epoch in range(num_epochs):`

---

## Checklist

- [ ] Can explain each training loop component
- [ ] Understand forward vs backward pass
- [ ] Know where attacks can occur in training
- [ ] Can implement complete training loop
- [ ] Training patterns typed without errors
- [ ] Project completed with gradient monitoring
