# Week 4 Day 6: Security Applications

## Learning Objectives
- Build production-quality threat classifier
- Understand adversarial example intuition
- Explore model extraction risks
- Apply security mindset to neural networks

## Key Concept: Neural Networks as Security Tools and Targets

Neural networks are both:
1. **Tools** for security (threat detection, malware classification)
2. **Targets** for attackers (adversarial examples, model extraction)

This dual nature is central to AI Security.

## Building a Production Threat Classifier

### Requirements

A production classifier needs:
- Proper train/validation/test split
- Feature preprocessing pipeline
- Threshold tuning for operational needs
- Evaluation metrics (precision, recall, F1)
- Confidence calibration

### Complete Implementation

```python
import numpy as np

class ThreatClassifier:
    """Production-ready neural network for threat detection."""
    
    def __init__(self, input_size, hidden_sizes=[16, 8], 
                 learning_rate=0.1, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        
        # Build layers
        layer_sizes = [input_size] + hidden_sizes + [1]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
        
        # Preprocessing parameters (set during fit)
        self.X_mean = None
        self.X_std = None
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, sig_output):
        return sig_output * (1 - sig_output)
    
    def preprocess(self, X, fit=False):
        """Standardize features."""
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        return (X - self.X_mean) / self.X_std
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(len(self.weights)):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # ReLU for hidden layers, sigmoid for output
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = self.sigmoid(z)
            
            self.activations.append(current)
        
        return current
    
    def backward(self, y):
        m = len(y)
        n_layers = len(self.weights)
        
        # Output layer delta (sigmoid)
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.activations[-1])
        
        self.gradients_W = [None] * n_layers
        self.gradients_b = [None] * n_layers
        
        for i in range(n_layers - 1, -1, -1):
            # Gradient with L2 regularization
            self.gradients_W[i] = np.dot(self.activations[i].T, delta) / m + self.lambda_reg * self.weights[i]
            self.gradients_b[i] = np.mean(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
        
        # Total loss
        mse = np.mean((self.activations[-1] - y) ** 2)
        reg = sum(np.sum(W**2) for W in self.weights) * self.lambda_reg / 2
        return mse + reg
    
    def update(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients_W[i]
            self.biases[i] -= self.learning_rate * self.gradients_b[i]
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """Train the classifier."""
        X_scaled = self.preprocess(X, fit=True)
        
        for epoch in range(epochs):
            self.forward(X_scaled)
            loss = self.backward(y)
            self.update()
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss:.4f}")
    
    def predict_proba(self, X):
        """Return threat probability."""
        X_scaled = self.preprocess(X, fit=False)
        return self.forward(X_scaled)
    
    def predict(self, X, threshold=0.5):
        """Return binary prediction."""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate(self, X, y, threshold=0.5):
        """Return comprehensive metrics."""
        predictions = self.predict(X, threshold)
        
        TP = np.sum((y == 1) & (predictions == 1))
        TN = np.sum((y == 0) & (predictions == 0))
        FP = np.sum((y == 0) & (predictions == 1))
        FN = np.sum((y == 1) & (predictions == 0))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
        }
```

### Usage

```python
# Create and train
model = ThreatClassifier(input_size=2, hidden_sizes=[8, 4])
model.fit(X_train, y_train, epochs=1000)

# Evaluate
train_metrics = model.evaluate(X_train, y_train)
test_metrics = model.evaluate(X_test, y_test)

print(f"Train F1: {train_metrics['f1']:.2%}")
print(f"Test F1: {test_metrics['f1']:.2%}")

# Adjust threshold for high-security scenario
high_recall_metrics = model.evaluate(X_test, y_test, threshold=0.3)
print(f"High-recall mode - Recall: {high_recall_metrics['recall']:.2%}")
```

## Adversarial Examples: Intuition

### What Are Adversarial Examples?

Inputs specifically crafted to fool the model while appearing normal to humans.

**Example scenario:**
- Your threat classifier correctly flags malicious traffic
- Attacker adds tiny modifications to their traffic
- Modified traffic appears identical to human analyst
- Model now classifies it as benign

### Why Neural Networks Are Vulnerable

Neural networks make predictions based on linear combinations of features. Small changes to many features can accumulate into large prediction changes.

**The math:**
```
prediction = sigmoid(W1*x1 + W2*x2 + ... + Wn*xn)
```

If attacker can add small ε to each feature in the direction that decreases the output:
```
adversarial = sigmoid(W1*(x1-ε·sign(W1)) + W2*(x2-ε·sign(W2)) + ...)
```

The cumulative effect can flip the prediction.

### Fast Gradient Sign Method (FGSM) — Conceptual

```python
def generate_adversarial(model, X, y, epsilon=0.1):
    """
    Generate adversarial example using FGSM.
    
    1. Compute gradient of loss with respect to INPUT
    2. Move input in direction that maximizes loss
    """
    # Forward pass
    model.forward(X)
    
    # Compute gradient of loss with respect to input
    # (This requires backpropagating all the way to X)
    output_delta = (model.activations[-1] - y) * model.sigmoid_derivative(model.activations[-1])
    
    # Backprop through all layers to get input gradient
    delta = output_delta
    for i in range(len(model.weights) - 1, -1, -1):
        delta = np.dot(delta, model.weights[i].T)
        if i > 0:
            delta = delta * model.relu_derivative(model.z_values[i-1])
    
    input_gradient = delta
    
    # Perturb in the direction of the gradient (to maximize loss)
    perturbation = epsilon * np.sign(input_gradient)
    X_adversarial = X + perturbation
    
    return X_adversarial
```

### Why This Matters for Security

Traditional security assumes attackers can't easily find inputs that bypass detection. With gradient-based attacks:
- Attackers can compute exactly how to modify inputs
- Modifications can be imperceptible
- Attack generalizes to similar inputs

## Model Extraction Attacks

### What Is Model Extraction?

Attacker queries your model repeatedly and uses responses to train a replica model.

**Attack flow:**
1. Attacker sends inputs, observes outputs
2. Attacker builds training dataset from queries
3. Attacker trains surrogate model on this data
4. Surrogate model approximates your model

### Why It Matters

Once attacker has a copy of your model:
- They can craft adversarial examples offline
- They understand your detection logic
- Your model is no longer proprietary

### Simple Extraction Attack

```python
def extract_model(target_model, input_dim, n_queries=1000):
    """
    Extract a model by querying it.
    """
    # Generate random queries
    X_queries = np.random.randn(n_queries, input_dim)
    
    # Get target model's predictions
    y_queries = target_model.predict_proba(X_queries)
    
    # Train surrogate model on query results
    surrogate = ThreatClassifier(input_size=input_dim, hidden_sizes=[8])
    surrogate.fit(X_queries, y_queries, epochs=500, verbose=False)
    
    return surrogate

# Attack
surrogate_model = extract_model(original_model, input_dim=2, n_queries=500)

# Test fidelity (how well surrogate matches original)
test_inputs = np.random.randn(100, 2)
original_preds = original_model.predict(test_inputs)
surrogate_preds = surrogate_model.predict(test_inputs)
fidelity = np.mean(original_preds == surrogate_preds)
print(f"Surrogate fidelity: {fidelity:.2%}")
```

### Defenses Against Extraction

| Defense | Mechanism | Trade-off |
|---------|-----------|-----------|
| Query limits | Restrict API calls | Hurts legitimate users |
| Output perturbation | Add noise to predictions | Reduces accuracy |
| Watermarking | Embed detectable patterns | Complexity |
| Detection | Monitor for extraction patterns | False positives |

## Threat Model for ML Systems

### Attack Surface Analysis

| Component | Attack Type | Risk Level |
|-----------|-------------|------------|
| Training data | Poisoning | High |
| Model parameters | Direct manipulation | Critical |
| Inference inputs | Adversarial examples | High |
| Model outputs | Information leakage | Medium |
| Model file | Extraction, theft | High |

### Defense-in-Depth for ML

1. **Data layer:** Validate training data, detect anomalies
2. **Model layer:** Regularization, ensemble methods
3. **Inference layer:** Input validation, output calibration
4. **Deployment layer:** Access control, rate limiting
5. **Monitoring layer:** Detect distribution shift, adversarial queries

## Practical Security Considerations

### When Deploying Threat Detection

1. **Don't trust training accuracy** — always evaluate on held-out data
2. **Monitor for distribution shift** — attackers may change tactics
3. **Plan for adversarial inputs** — assume sophisticated attackers
4. **Protect the model** — it's now a target
5. **Have fallbacks** — ML is one layer, not the only layer

### Red Team Questions

When evaluating your ML security system:
- Can an attacker craft inputs that evade detection?
- Can they extract the model through the API?
- Can they poison future training data?
- What happens when the model is wrong?
- How would you detect model failure?

## Exercises

1. **Build production classifier:** Implement the ThreatClassifier with train/validation/test splits. Report all metrics.

2. **Threshold analysis:** Plot precision vs recall for thresholds from 0.1 to 0.9. Find the optimal threshold for different operational needs.

3. **Extraction simulation:** Train a model, then extract it with 100, 500, and 1000 queries. How does fidelity scale?

4. **Adversarial intuition:** Manually modify a correctly-classified threat sample. How much change is needed to flip the prediction?

## Key Takeaways

1. Production ML systems need proper evaluation, not just training accuracy
2. Neural networks are vulnerable to adversarial examples
3. Small, targeted input modifications can flip predictions
4. Model extraction is a real threat for deployed models
5. Defense-in-depth applies to ML systems
6. ML is one security layer, not a complete solution

## Security Principles for ML

```
1. Assume adversarial inputs
2. Protect the model as an asset
3. Monitor for unexpected behavior
4. Plan for model failure
5. Layer ML with traditional controls
```

## Next: Day 7

Review and integration — complete neural network pipeline, comparison to Week 3, and bridge to Week 5 (CNNs, RNNs).
