# Day 6: NumPy Introduction

## Learning Goals
- Understand arrays vs lists
- Perform vectorized operations (no loops)
- Use boolean masks for filtering
- Analyze data with NumPy statistics

## Core Concept

**NumPy operates on entire arrays at once.**

Python list (slow):
```python
doubled = []
for n in numbers:
    doubled.append(n * 2)
```

NumPy array (fast):
```python
doubled = data * 2
```

Same result. No loop. Faster on large data.

## Why NumPy Matters

- Security logs can have millions of entries
- ML algorithms process huge datasets
- Loops are slow; vectorized operations are fast
- NumPy is the foundation of all Python ML

## Creating Arrays

```python
import numpy as np

# From list
data = np.array([1, 2, 3, 4, 5])
print(data)        # [1 2 3 4 5]
print(type(data))  # <class 'numpy.ndarray'>
```

## Vectorized Operations

Operations apply to every element:

```python
data = np.array([10, 20, 30, 40, 50])

print(data + 5)     # [15 25 35 45 55]
print(data * 2)     # [20 40 60 80 100]
print(data / 10)    # [1. 2. 3. 4. 5.]
```

## Boolean Masks

Comparisons return arrays of True/False:

```python
data = np.array([10, 20, 30, 40, 50])
mask = data > 25
print(mask)  # [False False True True True]
```

Use masks to filter:

```python
print(data[mask])  # [30 40 50]
```

Or combine into one line:

```python
print(data[data > 25])  # [30 40 50]
```

### Security Application

```python
counts = np.array([1, 3, 7, 2, 15, 4, 8])
blocked = counts[counts >= 5]
print(blocked)  # [7 15 8] - IPs to block
```

## Statistics Functions

```python
counts = np.array([1, 3, 7, 2, 15, 4, 8])

print(np.sum(counts))   # 40 (total)
print(np.mean(counts))  # 5.71 (average)
print(np.max(counts))   # 15 (highest)
print(np.min(counts))   # 1 (lowest)
print(np.std(counts))   # standard deviation
```

## Finding Positions

`np.max()` returns the value. `np.argmax()` returns the **index**:

```python
counts = np.array([1, 3, 7, 2, 15, 4, 8])
print(np.max(counts))     # 15 (the value)
print(np.argmax(counts))  # 4 (position of 15)
```

## 2D Arrays

Real data has rows and columns:

```python
# Each row: [hour, attempts]
data = np.array([
    [0, 5],
    [1, 3],
    [2, 12],
    [3, 8]
])

print(data.shape)  # (4, 2) - 4 rows, 2 columns
```

### Slicing 2D Arrays

```python
data[row, column]

data[0, 0]   # First row, first column → 0
data[0, 1]   # First row, second column → 5
data[:, 0]   # ALL rows, first column → [0 1 2 3]
data[:, 1]   # ALL rows, second column → [5 3 12 8]
```

### Find Peak Hour

```python
attempts = data[:, 1]           # Get all attempt counts
peak_index = np.argmax(attempts) # Which row has max
print(data[peak_index])          # Full row for that hour
```

## Anomaly Detection

Find values far from normal:

```python
hourly_counts = np.array([2, 3, 45, 52, 48, 5, 4, 3])

average = np.mean(hourly_counts)    # ~20
threshold = average * 2              # 40

anomalies = hourly_counts > threshold
print(hourly_counts[anomalies])     # [45 52 48] - suspicious hours
```

### Using `np.where()` for Indices

```python
anomaly_indices = np.where(hourly_counts > threshold)[0]
print(anomaly_indices)  # [2 3 4] - which hours
```

## Key Concepts Learned

| Concept | Purpose |
|---------|---------|
| `np.array()` | Create array from list |
| `data * 2` | Vectorized math |
| `data > 25` | Vectorized comparison → boolean mask |
| `data[mask]` | Filter using boolean mask |
| `np.sum/mean/max/min` | Statistics |
| `np.argmax()` | Index of max value |
| `data.shape` | Dimensions of array |
| `data[:, 1]` | Slice column from 2D array |
| `np.where()` | Find indices where condition is true |

## Success Criteria

- [ ] Can create NumPy arrays
- [ ] Can perform vectorized operations
- [ ] Can filter arrays with boolean masks
- [ ] Can calculate statistics
- [ ] Can work with 2D arrays
- [ ] Built anomaly detection

## Common Mistakes

1. **Using loops:** NumPy's power is avoiding loops—use vectorized operations
2. **`max` vs `argmax`:** `max` = value, `argmax` = position
3. **Forgetting `[0]` in `np.where()`:** `np.where()` returns a tuple
4. **Wrong slice syntax:** `data[:, 1]` not `data[:,1]` (space helps readability)

## Next Day Preview

Day 7 combines everything into a complete security monitoring tool.
