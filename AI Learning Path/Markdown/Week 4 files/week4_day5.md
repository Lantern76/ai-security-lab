# Week 4 Day 5: Regularization

## Learning Objectives
- Understand why neural networks overfit
- Implement L2 regularization (weight decay)
- Understand dropout conceptually
- Apply regularization to threat detection network

## Key Concept: The Overfitting Problem

### Why Neural Networks Overfit

Neural networks have many parameters. More parameters = more capacity to:
- Learn complex patterns ✓
- Memorize training data ✗

**Overfitting:** The model learns the training data too well, including noise and idiosyncrasies that don't generalize.

### Signs of Overfitting

```
Training loss:  0.01 (very low)
Test loss:      0.35 (much higher)

Training accuracy:  99%
Test accuracy:      72%
```

The gap between training and test performance is the overfitting signal.

### Visual Intuition

**Underfitting:** Decision boundary is too simple (straight line for curved data)
**Good fit:** Boundary captures the true pattern
**Overfitting:** Boundary wraps around every training point, including outliers

## Regularization: The Solution

**Regularization** = techniques that constrain the model to prevent overfitting.

Key idea: Add penalty for model complexity, so the model finds the simplest solution that fits the data well.

## L2 Regularization (Weight Decay)

### The Intuition

Large weights create sharp, complex decision boundaries. By penalizing large weights, we encourage smoother, simpler boundaries.

### Mathematical Formulation

**Original loss:**
```
Loss = MSE(predictions, y)
```

**Regularized loss:**
```
Loss = MSE(predictions, y) + λ * Σ(weights²)
```

Where:
- λ (lambda) is the regularization strength
- Σ(weights²) is the sum of all squared weights

### Why "Weight Decay"?

The gradient of the regularization term is:
```
∂(λ * W²)/∂W = 2λW
```

This adds `2λW` to the gradient, which pulls weights toward zero each update. Hence "decay."

### Implementation

```python
class NeuralNetworkL2:
    def __init__(self, input_size, hidden_size, output_size, 
                 learning_rate=0.1, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, sig_output):
        return sig_output * (1 - sig_output)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.hidden = self.sigmoid(self.z1)
        self.z2 = np.dot(self.hidden, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)
        return self.output
    
    def backward(self, X, y):
        m = len(y)
        
        # Standard backprop
        output_error = self.output - y
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
        
        # Gradients WITH L2 regularization
        self.gradient_W2 = np.dot(self.hidden.T, output_delta) / m + self.lambda_reg * self.W2
        self.gradient_b2 = np.mean(output_delta, axis=0, keepdims=True)
        self.gradient_W1 = np.dot(X.T, hidden_delta) / m + self.lambda_reg * self.W1
        self.gradient_b1 = np.mean(hidden_delta, axis=0, keepdims=True)
        
        # Loss WITH regularization term
        mse_loss = np.mean(output_error ** 2)
        reg_loss = (self.lambda_reg / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        
        return mse_loss + reg_loss
    
    def update(self):
        self.W2 -= self.learning_rate * self.gradient_W2
        self.b2 -= self.learning_rate * self.gradient_b2
        self.W1 -= self.learning_rate * self.gradient_W1
        self.b1 -= self.learning_rate * self.gradient_b1
```

**Key changes:**
1. Gradients include `+ lambda_reg * W`
2. Loss includes `+ (lambda/2) * sum(W²)`
3. Biases are NOT regularized (standard practice)

### Choosing Lambda

| Lambda | Effect |
|--------|--------|
| 0 | No regularization |
| 0.001 | Light regularization |
| 0.01 | Moderate regularization |
| 0.1 | Strong regularization |
| 1.0 | Very strong (may underfit) |

**Too low:** Doesn't prevent overfitting
**Too high:** Forces weights too small, underfits

### Experiment

```python
lambdas = [0, 0.001, 0.01, 0.1]

for lam in lambdas:
    model = NeuralNetworkL2(input_size=2, hidden_size=8, output_size=1,
                            lambda_reg=lam)
    model.fit(X_train, y_train, epochs=1000, verbose=False)
    
    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)
    gap = train_acc - test_acc
    
    print(f"λ={lam}: Train={train_acc:.2%}, Test={test_acc:.2%}, Gap={gap:.2%}")
```

## Dropout (Conceptual)

### The Intuition

During training, randomly "drop" (zero out) neurons with probability p.

**Why it works:**
- Prevents neurons from co-adapting
- Forces redundant representations
- Acts like training many smaller networks
- At test time, use all neurons (scaled)

### How It Works

**Training:**
```python
# Hidden layer with dropout
hidden = sigmoid(z1)
dropout_mask = np.random.binomial(1, 1-p, hidden.shape) / (1-p)
hidden = hidden * dropout_mask
```

**Testing:**
```python
# No dropout at test time — use all neurons
hidden = sigmoid(z1)
```

### The Scaling Factor

We divide by `(1-p)` during training so the expected value stays the same:
- If p=0.5, half the neurons are zeroed
- Remaining neurons are scaled by 2x
- Expected output magnitude stays constant

### Dropout Rate

| Rate | Effect |
|------|--------|
| 0.0 | No dropout |
| 0.2 | Light dropout (common) |
| 0.5 | Standard dropout |
| 0.8 | Heavy dropout (risky) |

### When to Use Dropout

- Large networks with overfitting risk
- When you have limited training data
- NOT typically on small networks or output layer

## L1 vs L2 Regularization

### L1 Regularization

```
Loss = MSE + λ * Σ|weights|
```

**Effect:** Drives some weights to exactly zero (sparsity)
**Use case:** Feature selection

### L2 Regularization

```
Loss = MSE + λ * Σ(weights²)
```

**Effect:** Drives all weights toward zero, but not exactly zero
**Use case:** General regularization

### Comparison

| Property | L1 | L2 |
|----------|----|----|
| Sparse weights | Yes | No |
| Gradient at 0 | Undefined | 0 |
| Computation | More complex | Simple |
| Common use | Feature selection | Weight decay |

## Early Stopping

### The Intuition

Monitor validation loss during training. Stop when validation loss starts increasing.

```
Epoch 100: Train=0.10, Val=0.15
Epoch 200: Train=0.05, Val=0.12  ← validation improving
Epoch 300: Train=0.02, Val=0.11  ← validation still improving
Epoch 400: Train=0.01, Val=0.13  ← validation getting worse!
Epoch 500: Train=0.005, Val=0.18 ← overfitting

→ Stop at Epoch 300
```

### Implementation

```python
def fit_with_early_stopping(self, X_train, y_train, X_val, y_val, 
                            epochs=1000, patience=50):
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(epochs):
        # Train
        self.forward(X_train)
        train_loss = self.backward(X_train, y_train)
        self.update()
        
        # Validate
        val_pred = self.forward(X_val)
        val_loss = np.mean((val_pred - y_val) ** 2)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best weights
            best_weights = (self.W1.copy(), self.b1.copy(), 
                           self.W2.copy(), self.b2.copy())
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            # Restore best weights
            self.W1, self.b1, self.W2, self.b2 = best_weights
            break
```

## Combining Regularization Techniques

In practice, you might combine:
- L2 regularization (always applicable)
- Dropout (for large networks)
- Early stopping (monitor validation)

```python
model = NeuralNetworkL2(
    input_size=2, 
    hidden_size=16, 
    output_size=1,
    lambda_reg=0.01,     # L2 regularization
    dropout_rate=0.2     # Dropout
)

model.fit_with_early_stopping(
    X_train, y_train,
    X_val, y_val,
    patience=50          # Early stopping
)
```

## Security Implications

### Regularization as Defense

**Against adversarial examples:**
- Smoother decision boundaries (from L2) are slightly more robust
- Not a strong defense, but helps

**Against model extraction:**
- Dropout during inference (keeping it on) adds noise to outputs
- Makes extraction harder but hurts accuracy

### Regularization Trade-offs for Security

| Stronger Regularization | Effect |
|------------------------|--------|
| Pro | Simpler model, harder to extract |
| Pro | Smoother boundaries, slightly more robust |
| Con | May miss subtle threat patterns |
| Con | Lower overall accuracy |

## Exercises

1. **L2 sweep:** Train networks with λ = [0, 0.001, 0.01, 0.1, 1.0]. Plot train/test accuracy vs lambda.

2. **Overfitting demonstration:** Train a large network (hidden=64) on small data (20 samples). Observe the train/test gap. Add L2 regularization and observe the change.

3. **Early stopping:** Implement early stopping. How many epochs does it run before stopping on the threat detection data?

4. **Weight analysis:** After training with and without L2, compare the distribution of weight values. Are regularized weights smaller?

## Key Takeaways

1. Overfitting = model memorizes training data instead of learning patterns
2. L2 regularization penalizes large weights, encouraging simplicity
3. Dropout randomly zeros neurons, preventing co-adaptation
4. Early stopping halts training when validation loss increases
5. Lambda controls regularization strength — tune via validation
6. Regularization slightly improves robustness but isn't a security solution
7. Combine techniques: L2 + dropout + early stopping

## Key Formulas

**L2 Regularized Loss:**
```
Loss = (1/n) * Σ(pred - y)² + (λ/2) * Σ(W²)
```

**L2 Regularized Gradient:**
```
∂Loss/∂W = normal_gradient + λ * W
```

**Dropout (training):**
```
mask = Bernoulli(1-p)
hidden = activation * mask / (1-p)
```

## Next: Day 6

Security applications — neural network for threat classification, adversarial example intuition, and model extraction risks.
