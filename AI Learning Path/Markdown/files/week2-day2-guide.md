# Week 2, Day 2: Matrices - Operating on Many Vectors

## Learning Goals
- Understand matrices as collections of vectors
- Perform matrix operations
- Use matrix multiplication for transformations
- Build intuition for how ML uses matrices

## Core Concept

**A matrix is a grid of numbers - many vectors stacked together.**

```python
# 3 login attempts, each with 4 features
# [hour, failed_count, seconds_since_last, is_new_ip]
data = np.array([
    [3, 5, 120, 1],   # attempt 1
    [14, 1, 3600, 0], # attempt 2  
    [3, 8, 45, 1]     # attempt 3
])
```

Each row = one data point (vector)
Each column = one feature across all points

## Security Analogy

Your security log is a matrix:
- Each row = one event
- Each column = one attribute (IP, port, timestamp, etc.)

ML processes the entire matrix at once, not row by row.

## Matrix Basics

### Creating Matrices

```python
import numpy as np

# From nested lists
data = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(data.shape)  # (2, 3) - 2 rows, 3 columns
print(data[0])     # [1 2 3] - first row
print(data[:, 0])  # [1 4] - first column
```

### Shape Convention

```
(rows, columns) = (samples, features)
```

- 100 login attempts with 5 features each: `(100, 5)`
- 1000 network packets with 10 features each: `(1000, 10)`

### Accessing Elements

```python
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

data[0, 0]    # 1 (row 0, col 0)
data[1, 2]    # 6 (row 1, col 2)
data[0, :]    # [1 2 3] (entire row 0)
data[:, 1]    # [2 5 8] (entire column 1)
data[0:2, :]  # first 2 rows
```

## Matrix Operations

### Element-wise Operations

Same as vectors - applies to every element:

```python
data = np.array([[1, 2], [3, 4]])

print(data + 10)   # [[11 12] [13 14]]
print(data * 2)    # [[2 4] [6 8]]
print(data ** 2)   # [[1 4] [9 16]]
```

### Matrix Addition

Add corresponding elements (shapes must match):

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A + B)  # [[6 8] [10 12]]
```

### Transpose

Flip rows and columns:

```python
data = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(data.shape)    # (2, 3)
print(data.T.shape)  # (3, 2)
print(data.T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

## Matrix Multiplication (Critical for ML!)

**Not element-wise.** Dot products of rows and columns.

```python
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[5, 6], [7, 8]])  # (2, 2)

result = np.dot(A, B)  # or A @ B
print(result)
# [[19 22]
#  [43 50]]
```

### How It Works

Each element in result = dot product of A's row with B's column:
```
result[0,0] = A[0,:] · B[:,0] = [1,2] · [5,7] = 1*5 + 2*7 = 19
result[0,1] = A[0,:] · B[:,1] = [1,2] · [6,8] = 1*6 + 2*8 = 22
```

### Shape Rules

```
(m, n) @ (n, p) = (m, p)
```

Inner dimensions must match. Result has outer dimensions.

```python
A = np.array([[1, 2, 3]])  # (1, 3)
B = np.array([[4], [5], [6]])  # (3, 1)

print((A @ B).shape)  # (1, 1) - single number!
print(A @ B)  # [[32]] - this is the dot product!
```

## Why Matrix Multiplication Matters

### ML Prediction in One Line

```python
# 100 samples, 4 features each
X = np.random.randn(100, 4)

# Model weights: 4 features -> 1 output
weights = np.array([[0.5], [-0.3], [0.8], [0.1]])  # (4, 1)

# Predictions for ALL 100 samples at once
predictions = X @ weights  # (100, 4) @ (4, 1) = (100, 1)
```

No loops. The entire dataset processed in one operation.

### Neural Network Layer

```python
# Input: 100 samples, 10 features
X = np.random.randn(100, 10)

# Layer weights: 10 inputs -> 5 outputs
W = np.random.randn(10, 5)

# Forward pass
output = X @ W  # (100, 10) @ (10, 5) = (100, 5)
```

This is literally what happens inside neural networks.

## Project: Batch Threat Scoring

Score many events at once using matrix multiplication:

```python
import numpy as np

# 5 security events, each with 4 features
# [failed_logins, unique_ips, night_activity, new_account]
events = np.array([
    [0, 1, 0, 0],   # Normal user
    [3, 1, 0, 0],   # Some failures
    [10, 5, 1, 1],  # Suspicious!
    [1, 1, 0, 0],   # Normal
    [8, 3, 1, 0],   # Suspicious
])

# Threat weights (learned from past attacks)
# Higher weight = more indicative of attack
weights = np.array([
    [0.3],   # failed_logins contributes 0.3 per failure
    [0.2],   # unique_ips contributes 0.2 per IP
    [0.4],   # night_activity contributes 0.4
    [0.1],   # new_account contributes 0.1
])

# Score ALL events at once
scores = events @ weights
print("Threat scores:")
print(scores)

# Classify
threshold = 2.0
for i, score in enumerate(scores):
    status = "ALERT" if score[0] > threshold else "OK"
    print(f"Event {i}: {score[0]:.2f} - {status}")
```

Output:
```
Threat scores:
[[0.2]
 [1.1]
 [4.9]
 [0.5]
 [3.5]]
Event 0: 0.20 - OK
Event 1: 1.10 - OK
Event 2: 4.90 - ALERT
Event 3: 0.50 - OK
Event 4: 3.50 - ALERT
```

## Broadcasting (Bonus Concept)

NumPy automatically expands shapes when possible:

```python
data = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
row_means = np.array([2, 5, 4])           # (3,)

# Subtract mean from each column (broadcast across rows)
centered = data - row_means
print(centered)
# [[-1 -3 -1]
#  [ 2  0  2]]
```

## Key Concepts Learned

| Concept | Code | ML Usage |
|---------|------|----------|
| Matrix shape | `data.shape` | (samples, features) |
| Row access | `data[i, :]` | One sample |
| Column access | `data[:, j]` | One feature |
| Transpose | `data.T` | Flip dimensions |
| Matrix multiply | `A @ B` | Transform data, predictions |

## Success Criteria

- [ ] Can create and access matrices
- [ ] Understand shape conventions
- [ ] Can perform matrix multiplication
- [ ] Understand shape rules for multiplication
- [ ] Built batch threat scorer
- [ ] Can explain how ML uses matrices

## Common Mistakes

1. **Shape mismatch:** `(3, 4) @ (5, 2)` fails - inner dims don't match
2. **`*` vs `@`:** `A * B` is element-wise, `A @ B` is matrix multiplication
3. **Row vs column confusion:** `data[0]` is row 0, `data[:, 0]` is column 0
4. **Forgetting reshape:** Sometimes need `weights.reshape(-1, 1)` for correct shape

## Exercises

1. Create a (100, 5) matrix of random login features
2. Create a (5, 1) weight vector
3. Compute threat scores for all 100 samples in one operation
4. Find the indices of the top 10 highest-scoring samples

## Tomorrow Preview

Day 3: Linear transformations - what matrix multiplication really means geometrically
