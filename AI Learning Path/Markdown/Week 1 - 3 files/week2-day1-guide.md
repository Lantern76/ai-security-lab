# Week 2: Mathematical Foundations for ML

## Week Overview

**Theme:** Math as notation for ideas you already understand

**Learning Objectives:**
- Linear algebra (vectors, matrices) in ML context
- NumPy as mathematical notation system
- Pandas for data manipulation
- Statistical thinking for ML

**End Goal:** Mathematical intuition that clarifies ML concepts, not obscures them

---

# Day 1: Vectors - The Language of Data

## Learning Goals
- Understand vectors as data containers
- Perform vector operations (add, scale, dot product)
- Connect vectors to security concepts
- Build intuition before formulas

## Core Concept

**A vector is a list of numbers with meaning.**

```python
# This is just a list
[192, 168, 1, 50]

# This is a vector (IP address as 4 dimensions)
ip_vector = np.array([192, 168, 1, 50])
```

The difference? We can do **math** on vectors to find patterns.

## Security Analogy

Think of a network packet as a vector:
```
packet = [source_port, dest_port, packet_size, flags, ttl]
```

Each position is a **dimension**. Each packet is a **point** in 5-dimensional space.

Similar packets cluster together. Attacks look different from normal traffic.

**This is the foundation of ML for security.**

## Vector Operations

### Creating Vectors

```python
import numpy as np

# Feature vector for a login attempt
# [hour, failed_attempts, time_since_last, is_new_ip]
attempt = np.array([3, 5, 120, 1])

print(attempt.shape)  # (4,) - 4 dimensions
print(len(attempt))   # 4 elements
```

### Vector Addition

Adding vectors adds corresponding elements:

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

result = v1 + v2
print(result)  # [5 7 9]
```

**Security meaning:** Combining feature vectors, aggregating counts

### Scalar Multiplication

Multiply every element by a number:

```python
v = np.array([2, 4, 6])
result = v * 3
print(result)  # [6 12 18]
```

**Security meaning:** Normalizing, scaling features

### Vector Magnitude (Length)

How "big" is a vector?

```python
v = np.array([3, 4])
magnitude = np.sqrt(np.sum(v ** 2))  # sqrt(9 + 16) = 5
# Or use NumPy:
magnitude = np.linalg.norm(v)
print(magnitude)  # 5.0
```

**Security meaning:** How far is this data point from origin? How extreme?

### Dot Product (Most Important!)

Multiply corresponding elements, then sum:

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
print(dot)  # 32
```

**What it means:** How similar are two vectors?
- Large positive = pointing same direction (similar)
- Zero = perpendicular (unrelated)
- Large negative = pointing opposite (opposites)

### Similarity with Dot Product

```python
normal_traffic = np.array([80, 443, 1500, 64])   # port, port, size, ttl
current_packet = np.array([80, 443, 1480, 64])   # similar
attack_packet = np.array([31337, 4444, 65000, 1])  # different

# Normalize first (divide by magnitude)
def normalize(v):
    return v / np.linalg.norm(v)

normal_norm = normalize(normal_traffic)
current_norm = normalize(current_packet)
attack_norm = normalize(attack_packet)

print(np.dot(normal_norm, current_norm))  # ~0.99 (very similar)
print(np.dot(normal_norm, attack_norm))   # ~0.03 (very different)
```

This is **cosine similarity** - the foundation of many ML algorithms.

## Project: Threat Similarity Detector

Build a tool that compares network events to known attack patterns:

```python
import numpy as np

def normalize(v):
    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    return v / magnitude

def similarity(v1, v2):
    """Cosine similarity: 1 = identical, 0 = unrelated, -1 = opposite"""
    return np.dot(normalize(v1), normalize(v2))

# Known attack pattern: [high_port, high_port, large_size, low_ttl]
attack_signature = np.array([31337, 4444, 65000, 1])

# Incoming traffic samples
samples = [
    np.array([80, 443, 1500, 64]),      # Normal web
    np.array([22, 22, 500, 64]),         # Normal SSH
    np.array([31337, 4444, 64000, 2]),   # Suspicious!
    np.array([80, 80, 1200, 60]),        # Normal
]

print("Similarity to attack pattern:")
for i, sample in enumerate(samples):
    sim = similarity(sample, attack_signature)
    status = "ALERT!" if sim > 0.9 else "OK"
    print(f"Sample {i}: {sim:.3f} - {status}")
```

## Key Concepts Learned

| Concept | Math | Code | Security Meaning |
|---------|------|------|------------------|
| Vector | List of numbers | `np.array([...])` | Feature representation |
| Addition | v1 + v2 | `v1 + v2` | Combining features |
| Scaling | c * v | `c * v` | Normalization |
| Magnitude | \|\|v\|\| | `np.linalg.norm(v)` | How extreme? |
| Dot product | v1 · v2 | `np.dot(v1, v2)` | Similarity measure |

## Why This Matters for ML

Every ML algorithm works with vectors:
- **Input:** Data point = vector of features
- **Model:** Weights = vector of learned values
- **Prediction:** Dot product of input and weights

Understanding vectors = understanding ML at its core.

## Success Criteria

- [ ] Can create vectors with NumPy
- [ ] Can perform vector operations
- [ ] Understand dot product as similarity
- [ ] Built threat similarity detector
- [ ] Can explain vectors in security terms

## Common Mistakes

1. **Forgetting normalization:** Raw dot product depends on magnitude. Normalize first for pure similarity.
2. **Confusing `*` and `np.dot`:** `v1 * v2` = element-wise. `np.dot(v1, v2)` = dot product.
3. **Wrong shape:** `(4,)` is a 1D vector. `(4, 1)` is a 2D column. They behave differently.

## Exercises

1. Create vectors for 5 login attempts with features: [hour, failed_count, seconds_since_last]
2. Calculate the magnitude of each
3. Find which two attempts are most similar (highest dot product of normalized vectors)
4. Build a function that flags attempts similar to a known attack pattern

## Tomorrow Preview

Day 2: Matrices - operating on many vectors at once
