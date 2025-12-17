# Imports

import pandas as pd
import numpy as np

# Data
raw_data = {
    "src_ip": ["10.0.0.5", None, "10.0.0.5", "192.168.1.10", "10.0.0.5"],
    "dst_port": [443, 22, 80, 443, 22],
    "protocol": ["TCP", "TCP", "UDP", "TCP", "TCP"],
    "bytes_sent": [500, 1200, None, 45000, 800],
    "packets": [5, 12, 8, None, 6],
    "action": ["allow", "deny", "allow", "allow", "deny"],
}

df = pd.DataFrame(raw_data)
print("=== RAW DATA ===")
print(df)

# Step 2: Handle missing data (use median - resists outliers)
df["bytes_sent"] = df["bytes_sent"].fillna(df["bytes_sent"].median())
df["packets"] = df["packets"].fillna(df["packets"].median())
df = df.dropna(subset=["src_ip"])
print("\n=== AFTER CLEANING ===")
print(df)

# Step 3: Z-score scaling (preserves anomalies)
numeric_cols = ["bytes_sent", "packets"]

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std

print("\n=== AFTER SCALING ===")
print(df)

# Step 4: One-hot encoding (enables schema monitoring)
df = pd.get_dummies(df, columns=["protocol", "action"])
print("\n=== AFTER ENCODING ===")
print(df)

# Step 5: Select final features (numeric only)
feature_cols = [
    "dst_port",
    "bytes_sent",
    "packets",
    "protocol_TCP",
    "protocol_UDP",
    "action_allow",
    "action_deny",
]

X = df[feature_cols].astype(float)

# Step 6: Print final results
print("\n=== FINAL FEATURE MATRIX ===")
print(X)
print("\nShape:", X.shape)

# Security Note: Defend the source data (raw_data) above all else.
# Once poisoned data enters this pipeline, every step validates it as "normal."
