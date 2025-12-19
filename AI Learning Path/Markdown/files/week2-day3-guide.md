# Week 2, Day 3: Linear Transformations - What Matrix Multiplication Means

## Learning Goals
- Understand matrix multiplication as transformation
- Visualize how matrices change data
- Connect to ML concepts (feature transformation)
- Build geometric intuition

## Core Concept

**Matrix multiplication transforms data from one space to another.**

When you multiply `X @ W`:
- `X` = your data (input space)
- `W` = transformation rules
- Result = data in new space

## Security Analogy

Think of it like a security dashboard that transforms raw logs into risk scores:

```
Raw data: [failed_logins, bytes_sent, ports_scanned, time_of_day]
    ↓ Transformation matrix
Risk view: [authentication_risk, data_exfil_risk, recon_risk]
```

The transformation matrix encodes **what combinations of raw features matter** for each risk type.

## Geometric View

### 2D Example

```python
import numpy as np

# Original point
point = np.array([1, 0])

# Rotation matrix (90 degrees counterclockwise)
rotation = np.array([
    [0, -1],
    [1, 0]
])

# Transform
new_point = rotation @ point
print(new_point)  # [0, 1] - rotated 90 degrees!
```

The matrix rotated our point. Different matrices do different things:
- Rotate
- Scale
- Stretch
- Shear
- Project to lower dimensions

### Scaling Matrix

```python
# Scale x by 2, y by 0.5
scale = np.array([
    [2, 0],
    [0, 0.5]
])

point = np.array([1, 2])
scaled = scale @ point
print(scaled)  # [2, 1]
```

## Dimension Reduction (Critical for ML)

Transform from high dimensions to low dimensions:

```python
# Data: 4 features
data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])  # (3, 4) - 3 samples, 4 features

# Projection matrix: 4 features -> 2 features
projection = np.array([
    [1, 0],
    [0, 1],
    [0.5, 0.5],
    [0.5, -0.5]
])  # (4, 2)

# Transform
reduced = data @ projection  # (3, 4) @ (4, 2) = (3, 2)
print(reduced.shape)  # (3, 2) - now only 2 features!
```

This is the essence of dimensionality reduction in ML.

## Feature Transformation

### Raw Features → Meaningful Features

```python
# Raw security data
# [failed_logins, successful_logins, bytes_in, bytes_out]
raw_data = np.array([
    [5, 100, 1000, 500],
    [50, 10, 100, 50000],
    [2, 200, 2000, 1800],
])

# Transformation to meaningful features
# Column 1: login_failure_ratio = failed / (failed + successful)
# Column 2: bytes_ratio = bytes_out / bytes_in
transform = np.array([
    [1, 0],      # failed_logins contributes to col 1
    [-0.01, 0],  # successful_logins (negative for ratio)
    [0, -0.001], # bytes_in (denominator)
    [0, 0.001],  # bytes_out (numerator)
])

# This is simplified - real ratios need non-linear operations
# But the principle: matrices combine features linearly
```

## Why Linear Transformations Matter for ML

### Neural Network = Stack of Linear Transforms + Non-linearities

```python
# Simplified neural network layer
def linear_layer(X, W, b):
    """Linear transformation: Y = X @ W + b"""
    return X @ W + b

# Input: 100 samples, 10 features
X = np.random.randn(100, 10)

# Layer 1: 10 features -> 5
W1 = np.random.randn(10, 5)
b1 = np.random.randn(5)
hidden = linear_layer(X, W1, b1)  # (100, 5)

# Layer 2: 5 features -> 2 (output classes)
W2 = np.random.randn(5, 2)
b2 = np.random.randn(2)
output = linear_layer(hidden, W2, b2)  # (100, 2)
```

**The weights `W1` and `W2` are learned transformations.**

### What the Network Learns

Through training, the network learns:
- `W1`: How to combine input features into useful intermediate features
- `W2`: How to combine intermediate features into predictions

The matrices encode the patterns that distinguish classes.

## Project: Feature Transformer

Build a system that transforms raw security metrics into risk scores:

```python
import numpy as np

class FeatureTransformer:
    def __init__(self, input_dim, output_dim):
        # Random initialization (in real ML, these are learned)
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)
    
    def transform(self, X):
        """Apply linear transformation"""
        return X @ self.W + self.b
    
    def set_weights(self, W, b):
        """Manually set transformation weights"""
        self.W = W
        self.b = b

# Create transformer: 4 raw features -> 3 risk scores
transformer = FeatureTransformer(4, 3)

# Set meaningful weights (normally these would be learned)
# Rows = input features, Cols = output features
weights = np.array([
    [0.8, 0.1, 0.0],   # failed_logins -> mostly auth_risk
    [0.0, 0.7, 0.2],   # bytes_out -> mostly exfil_risk
    [0.1, 0.0, 0.9],   # ports_scanned -> mostly recon_risk
    [0.3, 0.2, 0.1],   # night_hours -> all risks slightly
])
biases = np.array([0.0, 0.0, 0.0])
transformer.set_weights(weights, biases)

# Sample data: [failed_logins, bytes_out, ports_scanned, night_hours]
events = np.array([
    [0, 1000, 0, 0],    # Normal: some data transfer
    [10, 500, 0, 1],    # Auth attack at night
    [0, 50000, 0, 0],   # Data exfiltration
    [2, 100, 100, 1],   # Reconnaissance
])

# Transform to risk scores
risk_scores = transformer.transform(events)

print("Risk Scores [Auth, Exfil, Recon]:")
print(risk_scores)

# Interpret results
labels = ["auth_risk", "exfil_risk", "recon_risk"]
for i, event_risks in enumerate(risk_scores):
    max_idx = np.argmax(event_risks)
    print(f"Event {i}: Primary risk = {labels[max_idx]} ({event_risks[max_idx]:.2f})")
```

## Composing Transformations

Multiple transformations = matrix multiplication chain:

```python
# Transform 1: raw -> intermediate
T1 = np.random.randn(4, 3)

# Transform 2: intermediate -> output  
T2 = np.random.randn(3, 2)

# Combined transformation
T_combined = T1 @ T2  # (4, 3) @ (3, 2) = (4, 2)

# These are equivalent:
data = np.random.randn(100, 4)
result1 = (data @ T1) @ T2
result2 = data @ T_combined
print(np.allclose(result1, result2))  # True
```

This is why neural networks work: each layer transforms, and the whole thing is one big learned transformation.

## Key Concepts Learned

| Concept | Meaning | ML Connection |
|---------|---------|---------------|
| Linear transform | Matrix changes data | Neural network layers |
| Dimension change | (n,m) @ (m,p) → (n,p) | Feature reduction/expansion |
| Composition | A @ B @ C | Stacked layers |
| Learned weights | Matrix values from training | What the model "knows" |

## Success Criteria

- [ ] Understand matrix multiplication as transformation
- [ ] Can change dimensions with matrices
- [ ] Built feature transformer class
- [ ] Can explain neural network layers as linear transforms
- [ ] Understand how matrices encode learned patterns

## Common Mistakes

1. **Order matters:** `A @ B ≠ B @ A` in general
2. **Bias forgotten:** Real layers are `X @ W + b`, not just `X @ W`
3. **Shape confusion:** Draw out dimensions when debugging

## Exercises

1. Create a rotation matrix for 45 degrees and apply to 2D points
2. Build a projection that reduces 10 features to 3
3. Chain two transformations and verify composition works
4. Modify FeatureTransformer to include a ReLU activation (hint: `np.maximum(0, x)`)

## Tomorrow Preview

Day 4: Pandas - practical data manipulation for real datasets
