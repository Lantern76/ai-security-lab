# Week 4 Day 1: Forward and Backward Pass

## Learning Objectives
- Understand why single-layer models are limited
- Implement forward propagation through two layers
- Implement backpropagation using the chain rule
- Train a neural network on threat detection data

## Key Concept: Why Multiple Layers?

### The Limitation of Linear Models

Logistic regression draws a single straight line (hyperplane) to separate classes. This works when data is **linearly separable** — when one clean boundary divides the classes.

**Problem scenario:** 
- Malicious traffic type A: high bytes AND high duration (data exfiltration)
- Malicious traffic type B: low bytes AND low duration (C2 beaconing)
- Normal traffic: sits in the middle

No single straight line can separate both threat types from normal traffic.

### The Neural Network Solution

Instead of branching decisions (like a decision tree), neural networks **curve the boundary**.

**The insight:**
1. Layer 1 transforms the input space (warps it)
2. Layer 2 draws a linear boundary in that transformed space
3. Linear in transformed space = curved in original space

**Analogy:** Crumple a paper with mixed colored dots. In 2D, no line separates them. In the crumpled 3D space, you might slice between them. The hidden layer is the "crumpling."

## Architecture

```
Input (2 features) → Hidden Layer (4 neurons) → Output (1 neuron)

Weight Matrices:
- W1: (2, 4) — connects input to hidden
- W2: (4, 1) — connects hidden to output

Data Flow:
X (100, 2) @ W1 (2, 4) → z1 (100, 4) → sigmoid → hidden (100, 4)
hidden (100, 4) @ W2 (4, 1) → z2 (100, 1) → sigmoid → output (100, 1)
```

## Matrix Shape Rules

Matrix multiplication `A @ B` requires:
- Inner dimensions match: `(m, n) @ (n, p)`
- Result shape: `(m, p)`

**Determining weight shapes:**
- W1 connects 2 inputs to 4 hidden neurons → shape (2, 4)
- W2 connects 4 hidden neurons to 1 output → shape (4, 1)

## The Forward Pass

```python
# Step 1: Input to hidden
z1 = np.dot(X, W1)      # (100, 2) @ (2, 4) → (100, 4)
hidden = sigmoid(z1)     # Apply nonlinearity

# Step 2: Hidden to output
z2 = np.dot(hidden, W2)  # (100, 4) @ (4, 1) → (100, 1)
output = sigmoid(z2)     # Final predictions
```

**Why sigmoid between layers?**

Without nonlinearity, two linear transformations collapse into one:
```
(X @ W1) @ W2 = X @ (W1 @ W2) = X @ W_combined
```

The nonlinearity prevents this collapse, giving depth its power.

## The Backward Pass (Backpropagation)

### The Problem

In logistic regression, error flows directly from output to input weights. With a hidden layer in between, how does the error reach W1?

### The Solution: Chain Rule

**Key insight:** We pass the error signal backward through the network.

1. Output error tells us how to adjust W2
2. Output error flows backward through W2 to become hidden error
3. Hidden error tells us how to adjust W1

### The Sigmoid Derivative

Sigmoid: `σ(z) = 1 / (1 + e^(-z))`

Derivative: `σ'(z) = σ(z) × (1 - σ(z))`

**Convenient property:** The derivative is expressed in terms of the sigmoid output, which we already computed during forward pass.

```python
def sigmoid_derivative(sig_output):
    return sig_output * (1 - sig_output)
```

### Backpropagation Steps

```python
# Step 1: Output layer gradient
output_delta = output_error * sigmoid_derivative(output)
gradient_W2 = np.dot(hidden.T, output_delta) / len(y)

# Step 2: Pass error backward through W2
hidden_error = np.dot(output_delta, W2.T)

# Step 3: Hidden layer gradient
hidden_delta = hidden_error * sigmoid_derivative(hidden)
gradient_W1 = np.dot(X.T, hidden_delta) / len(y)
```

**Shape verification:**
- gradient_W2: (4, 1) — matches W2
- gradient_W1: (2, 4) — matches W1

## Complete Training Loop

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(sig_output):
    return sig_output * (1 - sig_output)

# Initialize weights
W1 = np.random.randn(2, 4)
W2 = np.random.randn(4, 1)
learning_rate = 0.1

for epoch in range(1000):
    # Forward pass
    z1 = np.dot(X, W1)
    hidden = sigmoid(z1)
    z2 = np.dot(hidden, W2)
    output = sigmoid(z2)
    
    # Calculate error and loss
    output_error = output - y
    loss = np.mean(output_error ** 2)
    
    # Backward pass
    output_delta = output_error * sigmoid_derivative(output)
    gradient_W2 = np.dot(hidden.T, output_delta) / len(y)
    
    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden)
    gradient_W1 = np.dot(X.T, hidden_delta) / len(y)
    
    # Update weights
    W1 = W1 - learning_rate * gradient_W1
    W2 = W2 - learning_rate * gradient_W2
```

## Results: Random vs Learnable Data

**Random labels (no pattern):**
```
Epoch 0: loss = 0.28
Epoch 900: loss = 0.23  # Barely improved
```

**Threat detection data (learnable pattern):**
```
Epoch 0: loss = 0.44
Epoch 900: loss = 0.03  # Dropped 93%
```

The network finds patterns when they exist.

## Security Considerations

### More Parameters = More Attack Surface

| Model | Parameters |
|-------|------------|
| Logistic Regression | 2 |
| Neural Network (2→4→1) | 12 |

**Tradeoffs of increased parameters:**
- ✅ More expressive power
- ❌ Higher overfitting risk
- ❌ More gradients for adversaries to exploit
- ❌ Model extraction becomes more valuable
- ❌ Increased computational cost

**Key insight:** Complexity is attack surface.

## Exercises

1. **Shape tracing:** If input has 5 features and hidden layer has 10 neurons, what are W1 and W2 shapes?

2. **Forward pass:** Manually trace a single sample through the network.

3. **Gradient intuition:** If a hidden neuron has large weights in W2, does it get more or less "blame" for output errors?

4. **Experiment:** What happens to training if you remove the sigmoid from the hidden layer?

## Key Takeaways

1. Multiple layers enable non-linear decision boundaries
2. Sigmoid (or other nonlinearity) between layers is essential
3. Backpropagation uses chain rule to flow error backward
4. The sigmoid derivative uses the forward pass output (efficient)
5. Gradient shapes must match weight shapes
6. More parameters = more capacity but also more attack surface

## Next: Day 2

Wrap the neural network in a class structure, add bias terms, and implement proper train/test evaluation.
