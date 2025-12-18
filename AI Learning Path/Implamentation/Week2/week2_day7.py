import pandas as pd
import numpy as np

# ============================================
# STEP 1: RAW DATA (Simulated network logs)
# ============================================

raw_logs = pd.DataFrame(
    {
        "src_ip": [
            "10.0.0.1",
            "10.0.0.1",
            "192.168.1.5",
            None,
            "10.0.0.1",
            "attacker.evil",
            "attacker.evil",
            "10.0.0.2",
            "192.168.1.5",
            "10.0.0.2",
        ],
        "dst_port": [443, 22, 80, 443, 22, 22, 22, 443, 80, 22],
        "bytes": [1500, None, 2300, 1800, 1600, 5000, 5100, None, 2100, 1400],
        "duration": [30, 45, None, 35, 32, 3600, 3605, 40, 38, 42],
        "time_since_last": [None, 120, 45, 300, 118, 60, 60, 200, 52, 180],
        "protocol": [
            "HTTPS",
            "SSH",
            "HTTP",
            "HTTPS",
            "SSH",
            "SSH",
            "SSH",
            "HTTPS",
            "HTTP",
            "SSH",
        ],
    }
)

print("=== RAW LOGS ===")
print(raw_logs)

# ============================================
# STEP 2: DATA CLEANING
# ============================================

df = raw_logs.copy()

# Flag 1: Check null ratio before filling
null_ratio = df.isnull().sum() / len(df)
print("=== NULL RATIOS ===")
print(null_ratio)

# Drop rows with missing src_ip (can't analyze without identifier)
df = df.dropna(subset=["src_ip"])

# Fill numeric columns with median
df["bytes"] = df["bytes"].fillna(df["bytes"].median())
df["duration"] = df["duration"].fillna(df["duration"].median())
df["time_since_last"] = df["time_since_last"].fillna(df["time_since_last"].median())

print("\n=== AFTER CLEANING ===")
print(df)
# ============================================
# ============================================
# STEP 3: FEATURE ENGINEERING
# ============================================

# IP frequency: count occurrences of each IP
ip_counts = df["src_ip"].value_counts()
df["ip_frequency"] = df["src_ip"].map(ip_counts)

print("=== IP FREQUENCY ===")
print(df[["src_ip", "ip_frequency"]])

# Timing variance per IP: low variance = suspicious
timing_variance = df.groupby("src_ip")["time_since_last"].std()
df["timing_std"] = df["src_ip"].map(timing_variance)
# Fill NaN (IPs with only 1 connection have no std)
df["timing_std"] = df["timing_std"].fillna(0)

print("=== TIMING VARIANCE ===")
print(df[["src_ip", "time_since_last", "timing_std"]].drop_duplicates("src_ip"))

# Step 4: Z-score scaling
numeric_cols = ["bytes", "duration", "time_since_last", "ip_frequency", "timing_std"]
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std
print("=== AFTER SCALING ===")
print(df[numeric_cols])

print("/n === AFTER ENCODING ===")

# Step 5: One-hot encoding
df = pd.get_dummies(df, columns=["protocol"])
print(df.head())

# ============================================
# STEP 6: THREAT SCORING
# ============================================

# Known bad IPs (in real system, this is a threat intel feed)
bad_ips = ["attacker.evil", "malware.bad"]
df["is_known_bad"] = df["src_ip"].isin(bad_ips).astype(int)

# Anomaly flag: bytes OR duration > 1.5 std
df["anomaly_flag"] = ((df["bytes"] > 1.5) | (df["duration"] > 1.5)).astype(int)

# Timing suspicion: invert timing_std (low variance = high suspicion)
df["timing_suspicion"] = -df["timing_std"]  # Flip sign: negative becomes positive

# Calculate threat score with your weights
df["threat_score"] = (
    1 * df["is_known_bad"] + 1 * df["anomaly_flag"] + 3 * df["timing_suspicion"]
)

print("=== THREAT SCORES ===")
print(
    df[["src_ip", "is_known_bad", "anomaly_flag", "timing_suspicion", "threat_score"]]
)

# Step 7: Flag high threat
df["high_priority"] = df["threat_score"] > 2.0

# Filter to only flagged rows
flagged = df[df["high_priority"] == True]

# Sort by threat score descending
flagged = flagged.sort_values("threat_score", ascending=False)

# Print results
print("=== HIGH PRIORITY ALERTS ===")
print(flagged[["src_ip", "threat_score", "high_priority"]])
