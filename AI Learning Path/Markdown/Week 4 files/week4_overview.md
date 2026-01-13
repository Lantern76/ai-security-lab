# Week 4: Neural Network Theory

## Overview

Week 4 extends the supervised learning foundations from Week 3 into multi-layer neural networks. The core insight: the same gradient descent loop (predict → error → gradient → update) applies, but now chains through multiple layers via backpropagation.

## Prerequisites

Before starting Week 4, you should have:
- Implemented gradient descent from scratch
- Built linear and logistic regression
- Understood loss functions (MSE) and their gradients
- Implemented train/test splits and evaluation metrics
- Completed Week 3 Day 7 (proper evaluation methodology)

## Learning Objectives

By the end of Week 4, you will:
1. Understand why depth matters (non-linear decision boundaries)
2. Implement forward propagation through multiple layers
3. Derive and implement backpropagation using the chain rule
4. Build neural networks from scratch in NumPy
5. Understand optimization dynamics (learning rate, convergence)
6. Identify neural network attack surfaces from a security perspective

## Key Concepts

### Why Multiple Layers?

Single-layer models (logistic regression) can only draw linear decision boundaries. Real-world problems often require curved or complex boundaries.

**The insight:** 
- Layer 1 transforms the input space
- Layer 2 draws a linear boundary in that transformed space
- The linear boundary in transformed space = curved boundary in original space

Stack more layers → more complex transformations → more expressive boundaries.

### The Forward Pass

Data flows through the network:
```
Input → [Weights × Input + Bias] → Activation → Hidden
Hidden → [Weights × Hidden + Bias] → Activation → Output
```

Each layer:
1. Computes a weighted sum: `z = np.dot(input, W)`
2. Applies nonlinearity: `activation = sigmoid(z)`

The nonlinearity is critical — without it, multiple linear layers collapse into one.

### The Backward Pass (Backpropagation)

Error flows backward through the network:
```
Output Error → Output Gradient → Hidden Error → Hidden Gradient
```

The chain rule connects layers:
- "How much should W2 change?" → Depends on output error
- "How much should W1 change?" → Depends on hidden error, which depends on output error flowing backward through W2

### The Sigmoid Derivative

Sigmoid: `σ(z) = 1 / (1 + e^(-z))`

Derivative: `σ'(z) = σ(z) × (1 - σ(z))`

The derivative tells us how sensitive the output is to changes in z. We use it during backpropagation to determine how much error to pass backward.

## Architecture Notation

```
Input (n_features) → Hidden (n_hidden) → Output (n_outputs)

Weight shapes:
- W1: (n_features, n_hidden)
- W2: (n_hidden, n_outputs)

Matrix multiplication rules:
- (samples, features) @ (features, hidden) → (samples, hidden)
- (samples, hidden) @ (hidden, outputs) → (samples, outputs)
```

## Security Considerations

Neural networks introduce new attack surfaces beyond classical ML:

1. **Increased overfitting risk**: More parameters = more capacity to memorize training data
2. **Gradient-based attacks**: Adversaries can compute gradients to craft adversarial examples
3. **Hidden layer vulnerabilities**: Intermediate representations may leak information
4. **Optimization exploits**: Learning rate and convergence properties can be manipulated

## Daily Breakdown

### Day 1: Forward and Backward Pass
- Build 2-layer network from scratch
- Implement forward propagation
- Implement backpropagation
- Train on threat detection data

### Day 2: Network as a Class
- Wrap network in reusable class structure
- Add bias terms
- Implement proper train/test evaluation

### Day 3: Optimization Deep Dive
- Learning rate effects
- Momentum and adaptive learning rates
- Convergence diagnostics

### Day 4: Architecture Decisions
- Hidden layer size effects
- Multiple hidden layers
- Activation function choices (ReLU, tanh)

### Day 5: Regularization
- Overfitting in neural networks
- L2 regularization (weight decay)
- Dropout concept

### Day 6: Security Applications
- Neural network for threat classification
- Adversarial example intuition
- Model extraction risks

### Day 7: Review and Integration
- Complete neural network pipeline
- Compare to Week 3 logistic regression
- Bridge to Week 5 (CNNs, RNNs)

## Code Pattern: Basic Neural Network

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(sig_output):
    return sig_output * (1 - sig_output)

# Initialize
W1 = np.random.randn(n_features, n_hidden)
W2 = np.random.randn(n_hidden, n_outputs)

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1)
    hidden = sigmoid(z1)
    z2 = np.dot(hidden, W2)
    output = sigmoid(z2)
    
    # Loss
    error = output - y
    loss = np.mean(error ** 2)
    
    # Backward pass
    output_delta = error * sigmoid_derivative(output)
    gradient_W2 = np.dot(hidden.T, output_delta) / len(y)
    
    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden)
    gradient_W1 = np.dot(X.T, hidden_delta) / len(y)
    
    # Update
    W1 -= learning_rate * gradient_W1
    W2 -= learning_rate * gradient_W2
```

## Key Formulas

**Forward Pass:**
- `z1 = X @ W1`
- `hidden = sigmoid(z1)`
- `z2 = hidden @ W2`
- `output = sigmoid(z2)`

**Backward Pass:**
- `output_delta = error × sigmoid'(output)`
- `gradient_W2 = hidden.T @ output_delta`
- `hidden_error = output_delta @ W2.T`
- `hidden_delta = hidden_error × sigmoid'(hidden)`
- `gradient_W1 = X.T @ hidden_delta`

**Update:**
- `W1 = W1 - lr × gradient_W1`
- `W2 = W2 - lr × gradient_W2`

## Common Mistakes

1. **Shape mismatches**: Always verify matrix dimensions before multiplying
2. **Forgetting activation**: Without nonlinearity, layers collapse
3. **Wrong derivative**: sigmoid_derivative takes sigmoid OUTPUT, not z
4. **Not scaling data**: Neural networks are sensitive to input scale
5. **Learning rate too high**: Causes oscillation or divergence
6. **Learning rate too low**: Training takes forever

## Success Criteria

By end of Week 4:
- [ ] Can derive backpropagation on paper
- [ ] Can implement neural network from scratch
- [ ] Understand effect of architecture choices
- [ ] Can diagnose training problems
- [ ] Can explain neural network vulnerabilities
