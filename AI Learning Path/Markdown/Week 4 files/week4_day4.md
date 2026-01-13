# Week 4 Day 4: Architecture Decisions

## Learning Objectives
- Understand effect of hidden layer size
- Implement networks with multiple hidden layers
- Compare activation functions (sigmoid, tanh, ReLU)
- Learn architecture selection principles

## Key Concept: Network Capacity

### What is Capacity?

**Capacity** = the complexity of functions a network can learn.

More capacity means:
- Can fit more complex patterns
- Can also memorize noise (overfitting risk)

Capacity is controlled by:
- Number of hidden layers (depth)
- Neurons per layer (width)
- Type of activation functions

## Hidden Layer Size Effects

### Too Few Neurons (Underfitting)

```
Architecture: Input (2) → Hidden (1) → Output (1)
```

With only 1 hidden neuron, the network can barely do more than logistic regression.

**Symptoms:**
- High training loss
- High test loss
- Both train and test accuracy low

### Too Many Neurons (Overfitting Risk)

```
Architecture: Input (2) → Hidden (100) → Output (1)
```

With 100 hidden neurons for a simple problem, the network has massive capacity.

**Symptoms:**
- Very low training loss
- Higher test loss
- Train accuracy >> test accuracy

### Right-Sized

```
Architecture: Input (2) → Hidden (4-8) → Output (1)
```

For the threat detection problem, 4-8 hidden neurons is plenty.

**Symptoms:**
- Low training loss
- Similarly low test loss
- Train and test accuracy close

### Experiment Code

```python
hidden_sizes = [1, 2, 4, 8, 16, 32]

for h_size in hidden_sizes:
    model = NeuralNetwork(input_size=2, hidden_size=h_size, output_size=1)
    model.fit(X_train, y_train, epochs=1000, verbose=False)
    
    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)
    
    print(f"Hidden={h_size}: Train={train_acc:.2%}, Test={test_acc:.2%}")
```

## Multiple Hidden Layers (Depth)

### Why Go Deeper?

Each layer learns increasingly abstract representations:
- Layer 1: Low-level features (edges, simple patterns)
- Layer 2: Combinations of low-level features
- Layer 3: Higher-level concepts

**For security applications:**
- Layer 1: Byte patterns, timing features
- Layer 2: Behavior patterns
- Layer 3: Threat categories

### Implementation

```python
class DeepNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        """
        layer_sizes: list of layer sizes, e.g., [2, 8, 4, 1]
        """
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, sig_output):
        return sig_output * (1 - sig_output)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(self.n_layers):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current = self.sigmoid(z)
            self.activations.append(current)
        
        return current
    
    def backward(self, y):
        m = len(y)
        self.gradients_W = []
        self.gradients_b = []
        
        # Output layer error
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.activations[-1])
        
        # Backpropagate through each layer
        for i in range(self.n_layers - 1, -1, -1):
            grad_W = np.dot(self.activations[i].T, delta) / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            
            self.gradients_W.insert(0, grad_W)
            self.gradients_b.insert(0, grad_b)
            
            if i > 0:  # Don't compute delta for input layer
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
        
        return np.mean((self.activations[-1] - y) ** 2)
    
    def update(self):
        for i in range(self.n_layers):
            self.weights[i] -= self.learning_rate * self.gradients_W[i]
            self.biases[i] -= self.learning_rate * self.gradients_b[i]
```

### Depth vs Width

| Architecture | Parameters | Character |
|--------------|------------|-----------|
| 2 → 16 → 1 | 49 | Wide and shallow |
| 2 → 4 → 4 → 1 | 37 | Narrow and deep |
| 2 → 8 → 4 → 1 | 53 | Balanced |

**Wide networks:** Good at memorization
**Deep networks:** Good at learning hierarchical features

## Activation Functions

### Sigmoid

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Properties:**
- Output range: (0, 1)
- Smooth gradient everywhere
- Problem: Saturates at extremes (vanishing gradient)

**Use for:** Output layer in binary classification

### Tanh

```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(tanh_output):
    return 1 - tanh_output ** 2
```

**Properties:**
- Output range: (-1, 1)
- Zero-centered (unlike sigmoid)
- Still saturates at extremes

**Use for:** Hidden layers when you need bounded outputs

### ReLU (Rectified Linear Unit)

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

**Properties:**
- Output range: [0, ∞)
- No saturation for positive values
- Computationally efficient
- Problem: "Dead ReLU" — neurons that always output 0

**Use for:** Hidden layers in deep networks (most common choice)

### Leaky ReLU

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)
```

**Properties:**
- Small gradient for negative values (fixes dead ReLU)
- Output range: (-∞, ∞)

### Comparison

| Activation | Vanishing Gradient | Dead Neurons | Speed |
|------------|-------------------|--------------|-------|
| Sigmoid | Yes | No | Slow |
| Tanh | Yes | No | Slow |
| ReLU | No | Yes | Fast |
| Leaky ReLU | No | No | Fast |

### Choosing Activation Functions

**Hidden layers:**
- Default: ReLU
- If dead neurons: Leaky ReLU
- If bounded output needed: Tanh

**Output layer:**
- Binary classification: Sigmoid
- Multi-class: Softmax (Week 5)
- Regression: Linear (no activation)

## Network with ReLU

```python
class NeuralNetworkReLU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, sig_output):
        return sig_output * (1 - sig_output)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.hidden = self.relu(self.z1)  # ReLU for hidden layer
        self.z2 = np.dot(self.hidden, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)  # Sigmoid for output
        return self.output
    
    def backward(self, X, y):
        m = len(y)
        
        output_error = self.output - y
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.relu_derivative(self.z1)  # Note: use z1, not hidden
        
        self.gradient_W2 = np.dot(self.hidden.T, output_delta) / m
        self.gradient_b2 = np.mean(output_delta, axis=0, keepdims=True)
        self.gradient_W1 = np.dot(X.T, hidden_delta) / m
        self.gradient_b1 = np.mean(hidden_delta, axis=0, keepdims=True)
        
        return np.mean(output_error ** 2)
```

**Note:** ReLU derivative uses `z1` (pre-activation), not `hidden` (post-activation).

### He Initialization for ReLU

```python
self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
```

ReLU zeros out half the values, so we need larger initial weights to compensate.

## Architecture Selection Guidelines

### Start Simple

1. Begin with 1 hidden layer
2. Start with `hidden_size = (input_size + output_size) / 2`
3. Train and evaluate
4. Add complexity only if underfitting

### Rules of Thumb

| Problem Complexity | Architecture |
|-------------------|--------------|
| Simple (linear-ish) | 1 layer, few neurons |
| Moderate | 1-2 layers, 8-64 neurons |
| Complex | 2-4 layers, 64-256 neurons |
| Very complex | Deep networks, specialized architectures |

### For Security Applications

**Threat detection:**
- Input: Network features (maybe 10-50)
- Hidden: 1-2 layers, 16-64 neurons
- Output: Threat probability

**Malware classification:**
- Input: File features (maybe 100-1000)
- Hidden: 2-3 layers, 64-256 neurons
- Output: Malware family (multi-class)

## Security Considerations

### Architecture as Information Leakage

The architecture itself reveals:
- Problem complexity
- Feature importance (through weight magnitudes)
- Decision boundaries

Model extraction attacks try to recover this.

### Defensive Architecture Choices

- Use slightly larger networks than needed (harder to extract exactly)
- Add dropout (Week 4 Day 5) — makes extraction harder
- Ensemble multiple architectures

## Exercises

1. **Width experiment:** Compare hidden sizes [2, 4, 8, 16, 32, 64] on threat detection data. Plot train/test accuracy vs width.

2. **Depth experiment:** Compare [2→4→1], [2→4→4→1], [2→4→4→4→1]. Does deeper always mean better?

3. **Activation comparison:** Implement ReLU and compare training speed to sigmoid on the same problem.

4. **Dead ReLU detection:** With ReLU activation, what percentage of hidden neurons output 0 for all inputs after training?

## Key Takeaways

1. Hidden layer size controls capacity — match to problem complexity
2. Deeper networks learn hierarchical features
3. ReLU is default for hidden layers; sigmoid/softmax for output
4. Start simple, add complexity only if needed
5. Architecture reveals information about your problem
6. Monitor train/test gap to detect over/underfitting

## Next: Day 5

Regularization techniques — L2 regularization, dropout, and preventing overfitting.
