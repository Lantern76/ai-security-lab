# Week 4 Day 3: Optimization Deep Dive

## Learning Objectives
- Understand learning rate effects on training
- Implement momentum for faster convergence
- Explore adaptive learning rates (intro to Adam)
- Diagnose training problems from loss curves

## Key Concept: The Optimization Landscape

### What Gradient Descent Actually Does

Imagine standing on a hilly landscape in fog. You can only feel the slope under your feet. Gradient descent says: "Walk downhill."

- **Gradient:** The slope direction (steepest ascent)
- **Negative gradient:** Downhill direction
- **Learning rate:** Step size

### The Problem

Walking downhill doesn't guarantee reaching the lowest valley:
- **Local minima:** Small valleys that aren't the deepest
- **Saddle points:** Flat areas that slope up in some directions, down in others
- **Plateaus:** Large flat regions where gradients are tiny

## Learning Rate Effects

### Too High

```
Epoch 0: loss = 0.44
Epoch 100: loss = 0.52   # Getting worse!
Epoch 200: loss = 0.48   # Oscillating
Epoch 300: loss = 0.51
```

**What's happening:** Steps are so large you overshoot the minimum, bouncing back and forth across the valley.

### Too Low

```
Epoch 0: loss = 0.44
Epoch 100: loss = 0.43
Epoch 200: loss = 0.42
Epoch 300: loss = 0.41
... (thousands of epochs later)
Epoch 10000: loss = 0.15
```

**What's happening:** Steps are tiny. You're walking downhill, but incredibly slowly.

### Just Right

```
Epoch 0: loss = 0.44
Epoch 100: loss = 0.31
Epoch 200: loss = 0.12
Epoch 300: loss = 0.05
```

**What's happening:** Steps are large enough for progress but small enough for stability.

### Finding Good Learning Rates

**Rule of thumb:** Start with 0.1, 0.01, or 0.001 and adjust based on loss curve.

**Learning rate finder (advanced):** Gradually increase learning rate while training, find where loss starts increasing rapidly.

## Momentum

### The Problem with Vanilla Gradient Descent

In ravine-shaped loss landscapes (steep in one direction, shallow in another), vanilla gradient descent oscillates across the steep dimension while making slow progress along the shallow dimension.

### The Solution: Momentum

**Idea:** Keep track of previous gradient directions. If we've been going the same direction, accelerate. If we're oscillating, the oscillations cancel out.

```python
# Vanilla gradient descent
W = W - learning_rate * gradient

# Gradient descent with momentum
velocity = momentum * velocity - learning_rate * gradient
W = W + velocity
```

**Analogy:** A ball rolling downhill accumulates momentum. It can roll through small bumps and accelerate on consistent slopes.

### Implementation

```python
class NeuralNetworkWithMomentum:
    def __init__(self, input_size, hidden_size, output_size, 
                 learning_rate=0.1, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Initialize velocities (same shape as weights)
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
    
    def update(self):
        # Update velocities
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * self.gradient_W2
        self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * self.gradient_b2
        self.v_W1 = self.momentum * self.v_W1 - self.learning_rate * self.gradient_W1
        self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * self.gradient_b1
        
        # Update weights using velocities
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1
```

### Momentum Value

- **momentum = 0:** Vanilla gradient descent
- **momentum = 0.9:** Standard choice, good balance
- **momentum = 0.99:** More aggressive, can overshoot

## Adaptive Learning Rates

### The Intuition

Different parameters may need different learning rates:
- Parameters with large gradients might need smaller steps
- Parameters with small gradients might need larger steps

### RMSprop (Root Mean Square Propagation)

Track the average of squared gradients. Scale learning rate inversely.

```python
# Accumulate squared gradients
cache = decay * cache + (1 - decay) * gradient**2

# Scale learning rate
W = W - learning_rate * gradient / (np.sqrt(cache) + epsilon)
```

Parameters with consistently large gradients get smaller effective learning rates.

### Adam (Adaptive Moment Estimation)

Combines momentum AND adaptive learning rates.

```python
# First moment (momentum)
m = beta1 * m + (1 - beta1) * gradient

# Second moment (squared gradients)
v = beta2 * v + (1 - beta2) * gradient**2

# Bias correction (important early in training)
m_corrected = m / (1 - beta1**t)
v_corrected = v / (1 - beta2**t)

# Update
W = W - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
```

**Standard values:** beta1=0.9, beta2=0.999, epsilon=1e-8

### When to Use What

| Optimizer | Use Case |
|-----------|----------|
| SGD | Simple problems, educational |
| SGD + Momentum | Standard choice, good baseline |
| Adam | Default for deep learning, robust |

## Diagnosing Training Problems

### Loss Curve Patterns

**Healthy training:**
```
Loss decreases smoothly, then plateaus
```

**Learning rate too high:**
```
Loss oscillates wildly or increases
```

**Learning rate too low:**
```
Loss decreases very slowly, nearly flat
```

**Stuck in local minimum:**
```
Loss plateaus early at high value
```

**Overfitting:**
```
Train loss keeps decreasing, validation loss starts increasing
```

### Debugging Checklist

1. **Loss not decreasing at all?**
   - Check gradient calculation
   - Verify weight update is applied
   - Check for NaN values

2. **Loss exploding (going to infinity)?**
   - Learning rate too high
   - Gradient explosion (deep networks)
   - Data not normalized

3. **Loss oscillating?**
   - Learning rate too high
   - Reduce by factor of 10

4. **Loss stuck?**
   - Learning rate too low
   - Stuck at saddle point (try momentum)
   - Network too small for problem

## Experiment: Learning Rate Comparison

```python
learning_rates = [0.001, 0.01, 0.1, 1.0]

for lr in learning_rates:
    model = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, 
                          learning_rate=lr)
    
    losses = []
    for epoch in range(500):
        model.forward(X_train)
        loss = model.backward(X_train, y_train)
        model.update()
        losses.append(loss)
    
    print(f"LR={lr}: Final loss = {losses[-1]:.4f}")
```

## Security Considerations

### Optimization as Attack Surface

**Gradient manipulation attacks:**
- Adversary corrupts gradient computation
- Model converges to adversary-chosen minimum
- Results in targeted misclassifications

**Learning rate attacks:**
- If adversary controls learning rate, they can:
  - Set too high → model never converges
  - Set too low → model appears stuck, user increases → then adversary increases → instability

### Detecting Optimization Attacks

- Monitor loss curve for unexpected patterns
- Validate gradient magnitudes
- Use gradient clipping as defense

## Exercises

1. **Learning rate sweep:** Train the same network with learning rates [0.001, 0.01, 0.1, 0.5, 1.0]. Plot loss curves.

2. **Momentum comparison:** Compare training with momentum=0 vs momentum=0.9. Which converges faster?

3. **Adam implementation:** Implement Adam optimizer from scratch.

4. **Diagnostic challenge:** Given a loss curve that increases then decreases then oscillates, what's happening?

## Key Takeaways

1. Learning rate is the most important hyperparameter
2. Too high = oscillation/divergence, too low = slow progress
3. Momentum helps escape local minima and speeds convergence
4. Adam combines momentum + adaptive learning rates
5. Loss curves tell you what's happening during training
6. Optimization is an attack surface — adversaries can manipulate convergence

## Key Formulas

**Vanilla SGD:**
```
W = W - lr * gradient
```

**SGD with Momentum:**
```
v = momentum * v - lr * gradient
W = W + v
```

**Adam (simplified):**
```
m = β1 * m + (1 - β1) * gradient
v = β2 * v + (1 - β2) * gradient²
W = W - lr * m / (√v + ε)
```

## Next: Day 4

Architecture decisions — hidden layer size effects, multiple hidden layers, and activation function choices.
