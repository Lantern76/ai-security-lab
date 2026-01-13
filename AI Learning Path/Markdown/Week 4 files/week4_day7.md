# Week 4 Day 7: Review and Integration

## Learning Objectives
- Consolidate all Week 4 concepts
- Build complete neural network pipeline
- Compare Week 3 (logistic regression) vs Week 4 (neural network)
- Bridge to Week 5 (specialized architectures)

## Week 4 Concept Map

```
Neural Network Theory
├── Architecture
│   ├── Forward Pass (Day 1)
│   │   ├── Matrix multiplication: X @ W
│   │   ├── Activation functions: sigmoid, ReLU
│   │   └── Layer-by-layer data flow
│   ├── Backward Pass (Day 1)
│   │   ├── Chain rule for gradients
│   │   ├── Error propagation through layers
│   │   └── Gradient calculation per layer
│   └── Design Choices (Day 4)
│       ├── Width (neurons per layer)
│       ├── Depth (number of layers)
│       └── Activation functions
├── Training
│   ├── Class Structure (Day 2)
│   │   ├── Weight initialization
│   │   ├── Bias terms
│   │   └── Encapsulation
│   ├── Optimization (Day 3)
│   │   ├── Learning rate effects
│   │   ├── Momentum
│   │   └── Adam optimizer
│   └── Regularization (Day 5)
│       ├── L2 weight decay
│       ├── Dropout
│       └── Early stopping
└── Security (Day 6)
    ├── Adversarial examples
    ├── Model extraction
    └── Threat modeling for ML
```

## Complete Neural Network Pipeline

### Full Implementation

```python
import numpy as np

class NeuralNetwork:
    """Complete neural network with all Week 4 concepts."""
    
    def __init__(self, layer_sizes, learning_rate=0.1, lambda_reg=0.01, momentum=0.9):
        """
        Args:
            layer_sizes: List like [input_dim, hidden1, hidden2, ..., output_dim]
            learning_rate: Step size for gradient descent
            lambda_reg: L2 regularization strength
            momentum: Momentum coefficient
        """
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.momentum = momentum
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.velocities_W = []
        self.velocities_b = []
        
        for i in range(self.n_layers):
            # He initialization for ReLU layers
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)
            self.velocities_W.append(np.zeros_like(W))
            self.velocities_b.append(np.zeros_like(b))
        
        # Preprocessing parameters
        self.X_mean = None
        self.X_std = None
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
    
    # === Activation Functions ===
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, sig_output):
        return sig_output * (1 - sig_output)
    
    # === Preprocessing ===
    
    def preprocess(self, X, fit=False):
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0) + 1e-8
        return (X - self.X_mean) / self.X_std
    
    # === Forward Pass ===
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(self.n_layers):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # ReLU for hidden, sigmoid for output
            if i < self.n_layers - 1:
                current = self.relu(z)
            else:
                current = self.sigmoid(z)
            
            self.activations.append(current)
        
        return current
    
    # === Backward Pass ===
    
    def backward(self, y):
        m = len(y)
        
        # Output layer delta
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.activations[-1])
        
        self.gradients_W = []
        self.gradients_b = []
        
        for i in range(self.n_layers - 1, -1, -1):
            # Gradient with L2 regularization
            grad_W = np.dot(self.activations[i].T, delta) / m + self.lambda_reg * self.weights[i]
            grad_b = np.mean(delta, axis=0, keepdims=True)
            
            self.gradients_W.insert(0, grad_W)
            self.gradients_b.insert(0, grad_b)
            
            # Backpropagate delta (if not at input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
        
        # Calculate loss
        mse = np.mean((self.activations[-1] - y) ** 2)
        reg = sum(np.sum(W**2) for W in self.weights) * self.lambda_reg / 2
        return mse + reg
    
    # === Update Weights ===
    
    def update(self):
        for i in range(self.n_layers):
            # Momentum update
            self.velocities_W[i] = self.momentum * self.velocities_W[i] - self.learning_rate * self.gradients_W[i]
            self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * self.gradients_b[i]
            
            self.weights[i] += self.velocities_W[i]
            self.biases[i] += self.velocities_b[i]
    
    # === Training ===
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=1000, patience=50, verbose=True):
        """
        Train with optional early stopping.
        """
        X_train_scaled = self.preprocess(X_train, fit=True)
        X_val_scaled = self.preprocess(X_val) if X_val is not None else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            # Training step
            self.forward(X_train_scaled)
            train_loss = self.backward(y_train)
            self.update()
            
            self.history['train_loss'].append(train_loss)
            
            # Validation step
            if X_val_scaled is not None:
                val_pred = self.forward(X_val_scaled)
                val_loss = np.mean((val_pred - y_val) ** 2)
                self.history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [(W.copy(), b.copy()) for W, b in zip(self.weights, self.biases)]
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    # Restore best weights
                    for i, (W, b) in enumerate(best_weights):
                        self.weights[i] = W
                        self.biases[i] = b
                    break
            
            if verbose and epoch % 100 == 0:
                msg = f"Epoch {epoch}: train_loss = {train_loss:.4f}"
                if X_val_scaled is not None:
                    msg += f", val_loss = {val_loss:.4f}"
                print(msg)
    
    # === Prediction ===
    
    def predict_proba(self, X):
        X_scaled = self.preprocess(X)
        return self.forward(X_scaled)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)
    
    # === Evaluation ===
    
    def evaluate(self, X, y, threshold=0.5):
        preds = self.predict(X, threshold)
        
        TP = np.sum((y == 1) & (preds == 1))
        TN = np.sum((y == 0) & (preds == 0))
        FP = np.sum((y == 0) & (preds == 1))
        FN = np.sum((y == 1) & (preds == 0))
        
        accuracy = (TP + TN) / len(y)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
        }
```

### Usage Example

```python
# Create threat detection data
np.random.seed(42)
normal = np.column_stack([
    np.random.randn(50) * 1000 + 3000,
    np.random.randn(50) * 20 + 40
])
threats = np.column_stack([
    np.random.randn(50) * 1000 + 7000,
    np.random.randn(50) * 30 + 100
])
X = np.vstack([normal, threats])
y = np.array([[0]]*50 + [[1]]*50)

# Shuffle and split
indices = np.random.permutation(len(y))
split1, split2 = int(0.7 * len(y)), int(0.85 * len(y))

X_train = X[indices[:split1]]
y_train = y[indices[:split1]]
X_val = X[indices[split1:split2]]
y_val = y[indices[split1:split2]]
X_test = X[indices[split2:]]
y_test = y[indices[split2:]]

# Train
model = NeuralNetwork(
    layer_sizes=[2, 8, 4, 1],
    learning_rate=0.1,
    lambda_reg=0.01,
    momentum=0.9
)

model.fit(X_train, y_train, X_val, y_val, epochs=1000, patience=50)

# Evaluate
train_metrics = model.evaluate(X_train, y_train)
test_metrics = model.evaluate(X_test, y_test)

print(f"\nTrain F1: {train_metrics['f1']:.2%}")
print(f"Test F1: {test_metrics['f1']:.2%}")
print(f"Gap: {train_metrics['f1'] - test_metrics['f1']:.2%}")
```

## Week 3 vs Week 4 Comparison

| Aspect | Logistic Regression (Week 3) | Neural Network (Week 4) |
|--------|------------------------------|-------------------------|
| **Architecture** | Single layer | Multiple layers |
| **Decision boundary** | Linear (straight line) | Non-linear (curved) |
| **Parameters** | 2 (for 2 features) | 12+ (depends on architecture) |
| **Training** | Single gradient calculation | Backpropagation through layers |
| **Capacity** | Low | High (adjustable) |
| **Overfitting risk** | Low | Higher |
| **Interpretability** | High (weights = feature importance) | Lower (distributed representations) |
| **Adversarial vulnerability** | Exists but simpler | More attack surface |

### When to Use Which

**Use Logistic Regression when:**
- Data is linearly separable
- Interpretability is important
- Limited training data
- Fast inference required

**Use Neural Networks when:**
- Non-linear relationships exist
- Sufficient training data available
- Higher accuracy needed
- Complexity is acceptable

## Key Formulas Reference

### Forward Pass
```
z_l = a_{l-1} @ W_l + b_l
a_l = activation(z_l)
```

### Backward Pass
```
δ_output = (a_L - y) × σ'(a_L)
δ_l = (δ_{l+1} @ W_{l+1}.T) × activation'(z_l)
∂Loss/∂W_l = a_{l-1}.T @ δ_l
∂Loss/∂b_l = mean(δ_l, axis=0)
```

### With Regularization
```
∂Loss/∂W_l = (a_{l-1}.T @ δ_l) / m + λ × W_l
```

### With Momentum
```
v = momentum × v - lr × gradient
W = W + v
```

## Self-Assessment Checklist

### Conceptual Understanding
- [ ] Can explain why multiple layers enable non-linear boundaries
- [ ] Can derive backpropagation using chain rule
- [ ] Understand effect of learning rate on training
- [ ] Can explain regularization's purpose and mechanism
- [ ] Understand adversarial example intuition

### Implementation Skills
- [ ] Can implement forward pass through multiple layers
- [ ] Can implement backward pass with correct gradient shapes
- [ ] Can add bias terms correctly
- [ ] Can implement L2 regularization
- [ ] Can implement momentum

### Practical Skills
- [ ] Can diagnose training problems from loss curves
- [ ] Can tune hyperparameters (learning rate, architecture, regularization)
- [ ] Can properly split data for train/validation/test
- [ ] Can evaluate model with appropriate metrics
- [ ] Can identify overfitting and apply fixes

## Bridge to Week 5

### What's Coming

Week 5 introduces specialized architectures:
- **CNNs (Convolutional Neural Networks):** For spatial data (images, malware visualization)
- **RNNs (Recurrent Neural Networks):** For sequential data (logs, network traffic over time)
- **Transformers:** For attention-based processing (coming in Week 9 for LLM security)

### How Week 4 Prepares You

Everything from Week 4 applies:
- Forward/backward passes work the same way
- Optimization (learning rate, momentum, Adam) applies directly
- Regularization concepts transfer
- Security considerations amplify with larger models

The key addition in Week 5: **specialized layers** that exploit data structure.

## Exercises

1. **Full pipeline:** Build complete threat classifier with train/val/test splits, early stopping, and comprehensive evaluation.

2. **Architecture search:** Compare [2→4→1], [2→8→1], [2→4→4→1], [2→8→4→1] on threat data. Which performs best on test set?

3. **Hyperparameter sweep:** Grid search over learning_rate=[0.01, 0.1], lambda_reg=[0.001, 0.01], momentum=[0, 0.9].

4. **Visualization:** Plot training and validation loss curves. Identify overfitting visually.

5. **Security analysis:** For your best model, analyze: How many queries would extraction require? What epsilon makes adversarial examples?

## Key Takeaways from Week 4

1. **Multiple layers enable non-linear decision boundaries**
2. **Backpropagation chains gradients through layers using chain rule**
3. **Architecture (width, depth, activation) controls model capacity**
4. **Optimization (learning rate, momentum) controls training dynamics**
5. **Regularization (L2, dropout, early stopping) prevents overfitting**
6. **Neural networks are both security tools and security targets**
7. **Always evaluate on held-out test data**
8. **Defense-in-depth applies to ML systems**

## Week 4 Complete

You've built neural networks from scratch, understanding every component from forward pass through backpropagation to regularization. You understand both how to use them for security (threat detection) and how they can be attacked (adversarial examples, extraction).

Ready for Week 5: Specialized architectures for spatial and sequential data.
