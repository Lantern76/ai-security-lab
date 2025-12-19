import pandas as pd
import numpy as np

raw_logs = pd.DataFrame(
    {
        "src_ip": [
            "10.0.0.1",
            "10.0.0.1",
            None,
            "192.168.1.5",
            "10.0.0.1",
            "evil.hacker",
        ],
        "bytes": [1200, None, 1500, 1300, 1250, 50000],
        "duration": [30, 35, 40, None, 32, 3600],
        "protocol": ["HTTP", "SSH", "HTTP", "SSH", "HTTP", "SSH"],
    }
)

df = raw_logs.copy()

print("=== RAW DATA ===")
print(raw_logs)

# Step 1 Drop rows with missing src_ip
df = df.dropna(subset=["src_ip"])

# Step 2: fill bytes nulls with median
df["bytes"] = df["bytes"].fillna(df["bytes"].median())

# Step 3: fill duration nulls with median
df["duration"] = df["duration"].fillna(df["duration"].median())

# Step 4: Z-score scale bytes and duratino
numeric_cols = ["bytes", "duration"]

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std

# Step 5: One-hot encode protocol
df = pd.get_dummies(df, columns=["protocol"])

# Step 6: Add ip_frequency column
ip_counts = df.groupby("src_ip").size()
df["ip_frequency"] = df["src_ip"].map(ip_counts)

# Step 7: Calculate threat score
df["threat_score"] = df["bytes"] + df["duration"] + (0.5 * df["ip_frequency"])

# Step 8: Flag rows with threat_score > 2
flagged = df[df["threat_score"] > 2]
print(flagged)
