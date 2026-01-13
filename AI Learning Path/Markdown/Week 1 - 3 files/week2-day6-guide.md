# Week 2, Day 6: Data Preprocessing for ML

## Learning Goals
- Understand why preprocessing matters
- Normalize and standardize features
- Encode categorical variables
- Handle missing data properly
- Build preprocessing pipelines

## Core Concept

**ML algorithms are picky eaters. Raw data must be prepared.**

Problems with raw data:
- Different scales (age: 0-100, salary: 0-1,000,000)
- Non-numeric categories ("allow", "deny")
- Missing values
- Outliers that dominate

Preprocessing fixes these issues.

## Security Analogy

Think of it like evidence processing before forensic analysis:
- Normalize timestamps to UTC
- Encode protocol names to numbers
- Fill in missing packet sizes
- Remove corrupted records

Same data, but now analyzable.

## Feature Scaling

### Why Scale?

Many ML algorithms use distance calculations. If one feature has a much larger range, it dominates.

```python
# Without scaling:
# bytes: [100, 50000, 30000]  → Dominates
# port: [22, 443, 80]          → Ignored
```

### Normalization (Min-Max Scaling)

Scale to [0, 1] range:

```python
import numpy as np

def normalize(data):
    """Scale to [0, 1]"""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

bytes_data = np.array([100, 50000, 30000, 10000])
normalized = normalize(bytes_data)
print(normalized)  # [0.0, 1.0, 0.599, 0.198]
```

**When to use:** When you need bounded values, neural networks

### Standardization (Z-Score Normalization)

Scale to mean=0, std=1:

```python
def standardize(data):
    """Scale to mean=0, std=1"""
    return (data - np.mean(data)) / np.std(data)

standardized = standardize(bytes_data)
print(standardized)  # [-1.06, 1.51, 0.66, -0.29]
print(f"Mean: {np.mean(standardized):.2f}")  # ~0
print(f"Std: {np.std(standardized):.2f}")    # ~1
```

**When to use:** Most ML algorithms, especially when outliers exist

### Using Scikit-Learn

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Reshape for sklearn (needs 2D)
data = bytes_data.reshape(-1, 1)

# Min-Max
mm_scaler = MinMaxScaler()
normalized = mm_scaler.fit_transform(data)

# Standardization
std_scaler = StandardScaler()
standardized = std_scaler.fit_transform(data)
```

**Critical:** Fit on training data, transform both training and test:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn from training
X_test_scaled = scaler.transform(X_test)         # Apply same transformation
```

## Encoding Categorical Variables

ML needs numbers, not strings.

### Label Encoding (Ordinal Categories)

When categories have order:

```python
from sklearn.preprocessing import LabelEncoder

severity = ["low", "medium", "high", "medium", "low"]

le = LabelEncoder()
encoded = le.fit_transform(severity)
print(encoded)  # [1, 2, 0, 2, 1] - arbitrary numbers

# Better: manual mapping for meaningful order
severity_map = {"low": 0, "medium": 1, "high": 2}
encoded = [severity_map[s] for s in severity]
print(encoded)  # [0, 1, 2, 1, 0]
```

### One-Hot Encoding (Nominal Categories)

When categories have no order:

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

protocols = ["tcp", "udp", "icmp", "tcp", "udp"]

# Pandas method (easier)
df = pd.DataFrame({"protocol": protocols})
one_hot = pd.get_dummies(df, columns=["protocol"])
print(one_hot)
```

Output:
```
   protocol_icmp  protocol_tcp  protocol_udp
0              0             1             0
1              0             0             1
2              1             0             0
3              0             1             0
4              0             0             1
```

Each category becomes its own column.

### When to Use Which?

| Encoding | When | Example |
|----------|------|---------|
| Label | Ordinal categories | low/medium/high |
| One-Hot | Nominal categories | tcp/udp/icmp |
| Binary | Two categories | allow/deny → 0/1 |

## Handling Missing Data

### Identify Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "ip": ["10.0.0.1", "10.0.0.2", None, "10.0.0.4"],
    "bytes": [100, np.nan, 300, 400],
    "port": [22, 443, 80, np.nan]
})

print(df.isnull().sum())  # Count nulls per column
```

### Strategy 1: Remove

```python
# Remove rows with ANY missing value
df_clean = df.dropna()

# Remove rows with missing in specific columns
df_clean = df.dropna(subset=["bytes"])
```

**When to use:** Few missing values, missing completely at random

### Strategy 2: Impute (Fill In)

```python
# Fill with constant
df["port"].fillna(0, inplace=True)

# Fill with mean
df["bytes"].fillna(df["bytes"].mean(), inplace=True)

# Fill with median (robust to outliers)
df["bytes"].fillna(df["bytes"].median(), inplace=True)

# Forward fill (time series)
df["ip"].fillna(method="ffill", inplace=True)
```

### Scikit-Learn Imputer

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")  # or "median", "most_frequent"
X_imputed = imputer.fit_transform(X)
```

## Feature Engineering

Creating new features from existing ones.

### Interaction Features

```python
df["bytes_per_second"] = df["bytes"] / df["duration"]
df["port_is_common"] = df["port"].isin([22, 80, 443]).astype(int)
```

### Binning

```python
df["hour_category"] = pd.cut(
    df["hour"], 
    bins=[0, 6, 12, 18, 24],
    labels=["night", "morning", "afternoon", "evening"]
)
```

### Log Transform (For Skewed Data)

```python
# Bytes often has long tail
df["log_bytes"] = np.log1p(df["bytes"])  # log(1+x) to handle zeros
```

## Project: Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create sample security data
np.random.seed(42)
n = 1000

raw_data = pd.DataFrame({
    "bytes": np.random.exponential(1000, n),
    "duration": np.random.normal(30, 10, n),
    "port": np.random.choice([22, 80, 443, 3389], n),
    "protocol": np.random.choice(["tcp", "udp", "icmp"], n),
    "action": np.random.choice(["allow", "deny"], n),
    "hour": np.random.randint(0, 24, n)
})

# Add some missing values
raw_data.loc[np.random.choice(n, 50), "bytes"] = np.nan
raw_data.loc[np.random.choice(n, 30), "duration"] = np.nan

print("Before preprocessing:")
print(raw_data.head())
print(f"Missing values:\n{raw_data.isnull().sum()}")

# Define preprocessing steps
numeric_features = ["bytes", "duration", "hour"]
categorical_features = ["protocol"]
binary_features = ["action"]

# Preprocessing pipelines for each type
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("onehot", OneHotEncoder(drop="first", sparse_output=False))
])

# Combine into single transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Add binary encoding manually
raw_data["action_encoded"] = (raw_data["action"] == "deny").astype(int)

# Fit and transform
X = raw_data[numeric_features + categorical_features]
X_processed = preprocessor.fit_transform(X)

print(f"\nAfter preprocessing:")
print(f"Shape: {X_processed.shape}")
print(f"No missing values: {not np.isnan(X_processed).any()}")
print(f"Mean of numeric (should be ~0): {X_processed[:, :3].mean(axis=0)}")
```

## Preprocessing Checklist

Before ML, always:

- [ ] **Check for missing values:** `df.isnull().sum()`
- [ ] **Check data types:** `df.dtypes`
- [ ] **Check value distributions:** `df.describe()`, `df.hist()`
- [ ] **Identify categorical vs numeric:** Encode appropriately
- [ ] **Scale numeric features:** StandardScaler or MinMaxScaler
- [ ] **Check for outliers:** Z-score, IQR method
- [ ] **Create derived features:** If domain knowledge suggests them

## Key Concepts Learned

| Concept | When | Code |
|---------|------|------|
| Standardization | Most ML | `StandardScaler()` |
| Normalization | Neural nets, bounded | `MinMaxScaler()` |
| One-hot encoding | Nominal categories | `pd.get_dummies()` |
| Label encoding | Ordinal categories | Manual mapping |
| Imputation | Missing values | `SimpleImputer()` |
| Log transform | Skewed data | `np.log1p()` |

## Success Criteria

- [ ] Can standardize and normalize features
- [ ] Can encode categorical variables
- [ ] Can handle missing data
- [ ] Built complete preprocessing pipeline
- [ ] Understand when to use each technique

## Common Mistakes

1. **Data leakage:** Fitting scaler on test data. Always fit on train only.
2. **Forgetting to encode:** Leaving strings in data for ML
3. **Wrong encoding:** Using label encoding for nominal categories (implies order)
4. **Over-imputing:** Sometimes missing values are informative—create "is_missing" flag

## Exercises

1. Create a preprocessing pipeline for network traffic data
2. Compare model performance with and without scaling
3. Test different imputation strategies on your security log
4. Create 3 new features from existing columns

## Tomorrow Preview

Day 7: Integration project - complete ML-ready data pipeline
