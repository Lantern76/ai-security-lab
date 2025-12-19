# Week 2, Day 4: Pandas - Practical Data Manipulation

## Learning Goals
- Understand DataFrames as labeled matrices
- Load, inspect, and clean data
- Filter, group, and aggregate
- Prepare data for ML pipelines

## Core Concept

**Pandas = NumPy with labels.**

NumPy: Access by position (`data[0, 2]`)
Pandas: Access by name (`df["ip_address"]`, `df.loc["event_42"]`)

## Security Analogy

A DataFrame is your security log with column headers:

| timestamp | source_ip | dest_port | action | bytes |
|-----------|-----------|-----------|--------|-------|
| 2024-01-15 | 10.0.0.5 | 443 | allow | 1500 |
| 2024-01-15 | 192.168.1.1 | 22 | deny | 0 |

You query it like SQL, but in Python.

## Creating DataFrames

### From Dictionary

```python
import pandas as pd

data = {
    "ip": ["10.0.0.1", "10.0.0.1", "192.168.1.5", "10.0.0.1"],
    "port": [22, 443, 80, 22],
    "action": ["deny", "allow", "allow", "deny"],
    "bytes": [0, 1500, 2300, 0]
}

df = pd.DataFrame(data)
print(df)
```

Output:
```
           ip  port action  bytes
0     10.0.0.1    22   deny      0
1     10.0.0.1   443  allow   1500
2  192.168.1.5    80  allow   2300
3     10.0.0.1    22   deny      0
```

### From CSV File

```python
df = pd.read_csv("security_log.csv")
```

## Inspecting Data

```python
df.shape          # (rows, columns)
df.head()         # First 5 rows
df.tail()         # Last 5 rows
df.info()         # Column types, non-null counts
df.describe()     # Statistics for numeric columns
df.columns        # List of column names
df.dtypes         # Data type of each column
```

## Selecting Data

### Select Columns

```python
df["ip"]              # Single column (returns Series)
df[["ip", "port"]]    # Multiple columns (returns DataFrame)
```

### Select Rows by Position

```python
df.iloc[0]        # First row
df.iloc[0:3]      # First 3 rows
df.iloc[0, 2]     # First row, third column
```

### Select Rows by Label

```python
df.loc[0]                    # Row with index 0
df.loc[0:2, ["ip", "port"]]  # Rows 0-2, specific columns
```

## Filtering (The Most Important Operation)

### Boolean Mask

```python
# Find all denied actions
mask = df["action"] == "deny"
denied = df[mask]
print(denied)
```

### Compound Conditions

```python
# Denied AND port 22
ssh_blocked = df[(df["action"] == "deny") & (df["port"] == 22)]

# Port 22 OR port 443
common_ports = df[(df["port"] == 22) | (df["port"] == 443)]

# NOT denied
allowed = df[df["action"] != "deny"]
```

**Critical:** Use `&` and `|`, not `and` and `or`. Wrap each condition in parentheses.

### Filter with Methods

```python
# IPs that start with "10."
internal = df[df["ip"].str.startswith("10.")]

# Bytes greater than 1000
large_transfers = df[df["bytes"] > 1000]

# Multiple values
target_ports = df[df["port"].isin([22, 80, 443])]
```

## Grouping and Aggregation

**This is where Pandas shines for security analysis.**

### Count by Group

```python
# How many events per IP?
df.groupby("ip").size()
```

Output:
```
ip
10.0.0.1       3
192.168.1.5    1
dtype: int64
```

### Multiple Aggregations

```python
# Per IP: count events, sum bytes, count unique ports
summary = df.groupby("ip").agg({
    "port": "count",           # Number of events
    "bytes": "sum",            # Total bytes
    "action": "nunique"        # Unique actions
})
print(summary)
```

### Named Aggregations (Cleaner)

```python
summary = df.groupby("ip").agg(
    event_count=("port", "count"),
    total_bytes=("bytes", "sum"),
    unique_actions=("action", "nunique")
)
```

## Adding/Modifying Columns

```python
# Add new column
df["is_blocked"] = df["action"] == "deny"

# Conditional column
df["risk_level"] = df["bytes"].apply(
    lambda x: "high" if x > 5000 else "normal"
)

# From calculation
df["bytes_kb"] = df["bytes"] / 1024
```

## Sorting

```python
df.sort_values("bytes", ascending=False)  # Largest first
df.sort_values(["ip", "port"])            # Multiple columns
```

## Handling Missing Data

```python
df.isnull().sum()           # Count nulls per column
df.dropna()                 # Remove rows with any null
df.fillna(0)                # Replace nulls with 0
df["bytes"].fillna(df["bytes"].mean())  # Fill with mean
```

## Project: Security Log Analyzer

```python
import pandas as pd
import numpy as np

# Create sample security log
np.random.seed(42)
n = 1000

log = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=n, freq="T"),
    "source_ip": np.random.choice(
        ["10.0.0.1", "10.0.0.2", "192.168.1.50", "attacker.evil.com"],
        n, p=[0.3, 0.3, 0.3, 0.1]  # attacker is 10%
    ),
    "dest_port": np.random.choice([22, 80, 443, 3389, 445], n),
    "action": np.random.choice(["allow", "deny"], n, p=[0.7, 0.3]),
    "bytes": np.random.exponential(1000, n).astype(int)
})

# Analysis

# 1. Events per source IP
print("Events per IP:")
print(log.groupby("source_ip").size().sort_values(ascending=False))

# 2. Denied events by IP
denied = log[log["action"] == "deny"]
print("\nDenied events per IP:")
print(denied.groupby("source_ip").size().sort_values(ascending=False))

# 3. Most targeted ports
print("\nMost targeted ports:")
print(log.groupby("dest_port").size().sort_values(ascending=False))

# 4. Suspicious activity: High denied ratio
ip_summary = log.groupby("source_ip").agg(
    total_events=("action", "count"),
    denied_count=("action", lambda x: (x == "deny").sum()),
    total_bytes=("bytes", "sum")
)
ip_summary["denied_ratio"] = ip_summary["denied_count"] / ip_summary["total_events"]

print("\nIP Risk Summary:")
print(ip_summary.sort_values("denied_ratio", ascending=False))

# 5. Flag suspicious IPs (denied ratio > 0.5 AND more than 20 events)
suspicious = ip_summary[
    (ip_summary["denied_ratio"] > 0.3) & 
    (ip_summary["total_events"] > 50)
]
print("\nSuspicious IPs:")
print(suspicious)
```

## Pandas vs NumPy

| Task | Use NumPy | Use Pandas |
|------|-----------|------------|
| Pure math on arrays | ✓ | |
| ML model input | ✓ | |
| Labeled data | | ✓ |
| Filtering by condition | Both work | ✓ (cleaner) |
| Group-by operations | | ✓ |
| Reading CSV files | | ✓ |
| Missing data | | ✓ |

**Typical workflow:** Load with Pandas → Clean → Convert to NumPy for ML

```python
# Prepare for ML
features = df[["port", "bytes", "is_blocked"]].values  # NumPy array
```

## Key Concepts Learned

| Concept | Code | Purpose |
|---------|------|---------|
| DataFrame | `pd.DataFrame(dict)` | Labeled 2D data |
| Select column | `df["col"]` | Get one feature |
| Filter | `df[df["col"] > x]` | Subset rows |
| Group-by | `df.groupby("col").agg()` | Summarize by category |
| Sort | `df.sort_values("col")` | Order rows |

## Success Criteria

- [ ] Can create DataFrames from dicts and CSVs
- [ ] Can inspect data (shape, head, describe)
- [ ] Can filter with boolean conditions
- [ ] Can group and aggregate
- [ ] Built security log analyzer
- [ ] Understand when to use Pandas vs NumPy

## Common Mistakes

1. **`and` vs `&`:** Use `&` for DataFrame conditions
2. **Missing parentheses:** `(df["a"] > 1) & (df["b"] < 2)` - each condition needs parens
3. **Modifying original:** Use `df.copy()` if you need to preserve original
4. **Index confusion:** After filtering, index may not be 0, 1, 2... Use `reset_index()`

## Exercises

1. Load a CSV of network traffic (create one or find sample)
2. Find the top 10 source IPs by event count
3. Calculate the denial rate per destination port
4. Find all events in a specific hour
5. Create a "risk score" column based on multiple factors

## Tomorrow Preview

Day 5: Statistics for ML - distributions, correlation, hypothesis thinking
