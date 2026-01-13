# Week 4 Day 2: Neural Network Class with Bias

## Learning Objectives
- Wrap neural network in reusable class structure
- Understand and implement bias terms
- Apply train/test split methodology to neural networks
- Compare training vs test performance

## Key Concept: Why Bias Terms?

### The Limitation Without Bias

Without bias, every decision boundary must pass through the origin. The network can only learn patterns centered at zero.

**With bias:** The network can shift its decision boundary anywhere in the space.

### Mathematical View

Without bias:
```
z = X @ W
```

With bias:
```
z = X @ W + b
```

The bias `b` is added to every sample's weighted sum, shifting the activation threshold.

## Architecture with Bias

```
Input (2) → Hidden (4) → Output (1)

Parameters:
- W1: (2, 4)  — input to hidden weights
- b1: (4,)   — hidden layer bias
- W2: (4, 1) — hidden to output weights
- b2: (1,)   — output layer bias

Total: 8 + 4 + 4 + 1 = 17 parameters
```

## The NeuralNetwork Class

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, sig_output):
        return sig_output * (1 - sig_output)
    
    def forward(self, X):
        # Store intermediate values for backprop
        self.z1 = np.dot(X, self.W1) + self.b1
        self.hidden = self.sigmoid(self.z1)
        self.z2 = np.dot(self.hidden, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)
        return self.output
    
    def backward(self, X, y):
        m = len(y)
        
        # Output layer
        output_error = self.output - y
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        # Hidden layer
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
        
        # Gradients
        self.gradient_W2 = np.dot(self.hidden.T, output_delta) / m
        self.gradient_b2 = np.mean(output_delta, axis=0, keepdims=True)
        self.gradient_W1 = np.dot(X.T, hidden_delta) / m
        self.gradient_b1 = np.mean(hidden_delta, axis=0, keepdims=True)
        
        return np.mean(output_error ** 2)
    
    def update(self):
        self.W2 -= self.learning_rate * self.gradient_W2
        self.b2 -= self.learning_rate * self.gradient_b2
        self.W1 -= self.learning_rate * self.gradient_W1
        self.b1 -= self.learning_rate * self.gradient_b1
    
    def fit(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y)
            self.update()
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss:.4f}")
    
    def predict_proba(self, X):
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)
```

## Bias Gradient Calculation

**Why `np.mean(delta, axis=0)`?**

Each sample contributes to the bias gradient. We average across all samples (axis=0) to get the gradient for each bias term.

```python
# For hidden layer bias (4 neurons)
self.gradient_b1 = np.mean(hidden_delta, axis=0, keepdims=True)
# Shape: (1, 4)

# For output layer bias (1 neuron)  
self.gradient_b2 = np.mean(output_delta, axis=0, keepdims=True)
# Shape: (1, 1)
```

## Weight Initialization

### The Problem with Large Random Weights

Large initial weights can cause:
- Sigmoid saturation (outputs near 0 or 1)
- Vanishing gradients (sigmoid derivative near 0)
- Slow or stalled learning

### Solution: Scaled Initialization

```python
self.W1 = np.random.randn(input_size, hidden_size) * 0.5
```

Multiplying by 0.5 (or another small factor) keeps initial activations in the sensitive region of sigmoid.

### Bias Initialization

Start biases at zero:
```python
self.b1 = np.zeros((1, hidden_size))
```

This is standard practice — let the network learn appropriate bias values.

## Train/Test Evaluation

### The Process

```python
# 1. Split data
indices = np.arange(len(y))
np.random.shuffle(indices)
split = int(0.8 * len(y))

X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# 2. Scale using training statistics only
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# 3. Train on training data only
model = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
model.fit(X_train_scaled, y_train, epochs=1000)

# 4. Evaluate on both sets
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)

train_accuracy = np.mean(train_preds == y_train)
test_accuracy = np.mean(test_preds == y_test)
```

### Interpreting Results

| Scenario | Train Acc | Test Acc | Diagnosis |
|----------|-----------|----------|-----------|
| 95% | 93% | Good generalization |
| 99% | 70% | Overfitting |
| 60% | 58% | Underfitting |
| 85% | 87% | Normal variance |

## Complete Example

```python
import numpy as np

# Create threat detection data
np.random.seed(42)
normal_bytes = np.random.randn(50) * 1000 + 3000
normal_duration = np.random.randn(50) * 20 + 40
threat_bytes = np.random.randn(50) * 1000 + 7000
threat_duration = np.random.randn(50) * 30 + 100

X = np.column_stack([
    np.concatenate([normal_bytes, threat_bytes]),
    np.concatenate([normal_duration, threat_duration])
])
y = np.array([[0]] * 50 + [[1]] * 50)

# Shuffle and split
indices = np.arange(len(y))
np.random.shuffle(indices)
split = int(0.8 * len(y))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Scale
X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Train
model = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
model.fit(X_train_scaled, y_train, epochs=1000)

# Evaluate
train_acc = np.mean(model.predict(X_train_scaled) == y_train)
test_acc = np.mean(model.predict(X_test_scaled) == y_test)
print(f"Train accuracy: {train_acc:.2%}")
print(f"Test accuracy: {test_acc:.2%}")
```

## Security Considerations

### Bias as Attack Vector

Bias terms shift decision boundaries. An adversary who can manipulate bias values during training (e.g., through gradient manipulation) can shift the boundary to misclassify specific inputs.

### Information Leakage

The network's learned biases encode information about the training data distribution. Model extraction attacks can recover this information.

## Exercises

1. **Ablation study:** Train with and without bias terms. Compare final loss and accuracy.

2. **Initialization experiment:** Try different weight initialization scales (0.1, 0.5, 1.0, 2.0). What happens to training?

3. **Overfitting detection:** Reduce training data to 20 samples. What happens to the train/test gap?

4. **Parameter counting:** For a network with architecture (10 → 20 → 5 → 1), calculate total parameters including biases.

## Key Takeaways

1. Bias terms allow decision boundaries to shift away from origin
2. Initialize weights small, biases at zero
3. Store intermediate values during forward pass for use in backprop
4. Bias gradients are averaged across samples
5. Train/test evaluation is essential — training accuracy can be misleading
6. The gap between train and test accuracy indicates overfitting

## Next: Day 3

Explore optimization dynamics — learning rate effects, momentum, and convergence diagnostics.
