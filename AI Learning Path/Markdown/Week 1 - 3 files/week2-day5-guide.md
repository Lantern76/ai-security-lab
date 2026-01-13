# Week 2, Day 5: Statistics for ML

## Learning Goals
- Understand distributions and their security meaning
- Calculate and interpret correlation
- Think statistically about anomalies
- Build foundation for ML evaluation

## Core Concept

**Statistics describes patterns in data. ML exploits those patterns.**

If you understand the statistical properties of normal traffic, you can detect when something is abnormal.

## Security Analogy

Normal traffic has patterns:
- Login attempts cluster during work hours
- Packet sizes follow certain distributions
- Bytes transferred correlates with session duration

Attacks deviate from these patterns. Statistics quantifies "how different."

## Descriptive Statistics

### Central Tendency

Where's the "middle" of the data?

```python
import numpy as np
import pandas as pd

# Login attempts per hour
attempts = np.array([5, 3, 8, 12, 4, 6, 7, 150, 5, 4])

print(f"Mean: {np.mean(attempts):.2f}")     # Average (20.4) - pulled by outlier
print(f"Median: {np.median(attempts):.2f}") # Middle value (5.5) - robust to outliers
print(f"Mode: {pd.Series(attempts).mode().values}")  # Most common
```

**Security insight:** Mean vs median difference reveals outliers. If mean >> median, you have extreme values (potential attacks).

### Spread (Variability)

How scattered is the data?

```python
print(f"Std Dev: {np.std(attempts):.2f}")   # Average distance from mean
print(f"Variance: {np.var(attempts):.2f}")  # Std dev squared
print(f"Range: {np.max(attempts) - np.min(attempts)}")
print(f"IQR: {np.percentile(attempts, 75) - np.percentile(attempts, 25)}")
```

**Security insight:** High variance = inconsistent behavior (suspicious). Low variance = predictable (normal user).

### Percentiles

```python
print(f"25th percentile: {np.percentile(attempts, 25)}")
print(f"50th percentile: {np.percentile(attempts, 50)}")  # Same as median
print(f"75th percentile: {np.percentile(attempts, 75)}")
print(f"99th percentile: {np.percentile(attempts, 99)}")  # Near-maximum
```

**Security insight:** 99th percentile is useful for "normal range." Anything above is suspicious.

## Distributions

### Normal Distribution

Most natural phenomena cluster around a mean:

```python
import numpy as np

# Generate normal data
normal_data = np.random.normal(loc=100, scale=15, size=1000)
# loc = mean, scale = std dev

print(f"Mean: {np.mean(normal_data):.2f}")  # ~100
print(f"Std: {np.std(normal_data):.2f}")    # ~15
```

**68-95-99.7 Rule:**
- 68% within 1 std dev of mean
- 95% within 2 std devs
- 99.7% within 3 std devs

### Z-Score: How Many Standard Deviations Away?

```python
def z_score(value, data):
    return (value - np.mean(data)) / np.std(data)

# Normal traffic: mean=100, std=15
traffic = np.random.normal(100, 15, 1000)

# Check specific values
print(f"Value 100: z = {z_score(100, traffic):.2f}")  # ~0 (at mean)
print(f"Value 130: z = {z_score(130, traffic):.2f}")  # ~2 (unusual)
print(f"Value 200: z = {z_score(200, traffic):.2f}")  # ~6.7 (very anomalous!)
```

**Rule of thumb:** |z| > 3 is suspicious. |z| > 4 is almost certainly anomalous.

### Detecting Anomalies with Z-Scores

```python
def find_anomalies(data, threshold=3):
    """Find values more than 'threshold' std devs from mean"""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    return np.where(np.abs(z_scores) > threshold)[0]

# Example: login attempts with one attacker
attempts = np.array([5, 3, 8, 4, 6, 7, 5, 4, 150, 5, 3, 4])
anomaly_indices = find_anomalies(attempts, threshold=2)
print(f"Anomalies at indices: {anomaly_indices}")
print(f"Values: {attempts[anomaly_indices]}")
```

## Correlation

**Correlation measures how two variables move together.**

- +1 = perfect positive (both go up together)
- 0 = no relationship
- -1 = perfect negative (one up, other down)

```python
# Generate correlated data
np.random.seed(42)
bytes_sent = np.random.normal(1000, 200, 100)
session_duration = bytes_sent * 0.01 + np.random.normal(0, 2, 100)  # Correlated

# Calculate correlation
correlation = np.corrcoef(bytes_sent, session_duration)[0, 1]
print(f"Correlation: {correlation:.3f}")  # High positive

# Uncorrelated
random_data = np.random.normal(0, 1, 100)
uncorr = np.corrcoef(bytes_sent, random_data)[0, 1]
print(f"Uncorrelated: {uncorr:.3f}")  # Near 0
```

### Security Application: Feature Correlation

```python
import pandas as pd

# Security features
df = pd.DataFrame({
    "failed_logins": np.random.poisson(3, 100),
    "unique_ips": np.random.poisson(2, 100),
    "bytes_out": np.random.exponential(1000, 100),
    "night_activity": np.random.binomial(1, 0.2, 100)
})

# Correlation matrix
print(df.corr())
```

**Why it matters:**
- Highly correlated features are redundant (pick one)
- Unexpected correlations might reveal attack patterns
- Feature engineering: combine correlated features

## Probability Distributions in Security

### Poisson Distribution: Event Counts

"How many events in a time window?"

```python
from scipy import stats

# Average 5 login failures per hour
lambda_param = 5

# Probability of exactly 10 failures?
prob_10 = stats.poisson.pmf(10, lambda_param)
print(f"P(X=10): {prob_10:.4f}")

# Probability of MORE than 10 failures?
prob_more_10 = 1 - stats.poisson.cdf(10, lambda_param)
print(f"P(X>10): {prob_more_10:.4f}")  # Very low if normal
```

**Security use:** If P(X > observed) < 0.01, it's a 1% chance event. Probably an attack.

### Exponential Distribution: Time Between Events

"How long until next event?"

```python
# Average 10 seconds between packets
rate = 1/10  # lambda = 1/mean

# Probability of waiting more than 30 seconds?
prob_wait = 1 - stats.expon.cdf(30, scale=1/rate)
print(f"P(wait > 30s): {prob_wait:.4f}")
```

## Project: Statistical Anomaly Detector

```python
import numpy as np
import pandas as pd

class StatisticalAnomalyDetector:
    def __init__(self):
        self.baseline_stats = {}
    
    def fit(self, data, column):
        """Learn baseline statistics from normal data"""
        values = data[column].values
        self.baseline_stats[column] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "p25": np.percentile(values, 25),
            "p75": np.percentile(values, 75),
            "p99": np.percentile(values, 99)
        }
        return self
    
    def score(self, data, column):
        """Return anomaly scores (z-scores)"""
        stats = self.baseline_stats[column]
        values = data[column].values
        z_scores = (values - stats["mean"]) / stats["std"]
        return np.abs(z_scores)
    
    def detect(self, data, column, threshold=3):
        """Find anomalous rows"""
        scores = self.score(data, column)
        return data[scores > threshold].copy()

# Generate normal traffic
np.random.seed(42)
normal_traffic = pd.DataFrame({
    "bytes": np.random.normal(1000, 200, 500),
    "duration": np.random.normal(30, 10, 500)
})

# Fit detector on normal data
detector = StatisticalAnomalyDetector()
detector.fit(normal_traffic, "bytes")

# New data with anomalies
test_traffic = pd.DataFrame({
    "bytes": np.concatenate([
        np.random.normal(1000, 200, 95),  # Normal
        np.array([5000, 6000, 7000, 8000, 9000])  # Anomalies
    ]),
    "duration": np.random.normal(30, 10, 100)
})

# Detect
anomalies = detector.detect(test_traffic, "bytes", threshold=3)
print(f"Found {len(anomalies)} anomalies:")
print(anomalies)
```

## Key Statistical Concepts for ML

| Concept | Formula/Code | ML Relevance |
|---------|--------------|--------------|
| Mean | `np.mean(x)` | Expected value |
| Std Dev | `np.std(x)` | Spread, normalization |
| Z-score | `(x - mean) / std` | Anomaly detection, normalization |
| Correlation | `np.corrcoef(x, y)` | Feature selection |
| Percentile | `np.percentile(x, p)` | Thresholds, outliers |

## Success Criteria

- [ ] Can calculate mean, std, percentiles
- [ ] Understand z-scores for anomaly detection
- [ ] Can compute and interpret correlation
- [ ] Built statistical anomaly detector
- [ ] Can think probabilistically about events

## Common Mistakes

1. **Mean without median:** Always check both to understand outlier impact
2. **Ignoring distribution shape:** Not all data is normal
3. **Correlation ≠ causation:** Two things moving together doesn't mean one causes the other
4. **Hard thresholds:** Don't use fixed rules; adapt to your data's distribution

## Exercises

1. Calculate z-scores for all IPs' failed login counts. Flag any with |z| > 2.
2. Find correlation between "bytes sent" and "session duration"
3. Use Poisson to calculate probability of seeing 20+ events if normal is 5/hour
4. Build a detector that flags IPs above 99th percentile in any metric

## Tomorrow Preview

Day 6: Data preprocessing - normalization, encoding, and feature engineering
