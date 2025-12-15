import numpy as np
import pandas as pd
# ============================================
# LOGIN ATTEMPT DATA (24 hours)
# ============================================

attempts = np.array(
    [
        12,
        15,
        8,
        10,
        14,
        11,
        9,
        13,
        16,
        10,
        12,
        14,
        11,
        10,
        15,
        12,
        9,
        11,
        13,
        10,
        14,
        12,
        150,
        11,
    ]
)

print("Data:", attempts)
print("Length:", len(attempts))

# ============================================
# BASIC STATISTICS
# ============================================

mean = np.mean(attempts)
median = np.median(attempts)
std = np.std(attempts)

print(f"\nMean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Standard deviation: {std:.2f}")

# ============================================
# Z-SCORES
# ============================================

z_scores = (attempts - mean) / std
print("\nZ-scores:")
print(z_scores.round(2))

# ============================================
# FIND ANOMALIES
# ============================================

threshold = 3
anomalies = np.where(np.abs(z_scores) > threshold)[0]

print(f"\nAnomaly at hours: {anomalies}")
print(f"Values: {attempts[anomalies]}")

# ============================================
# ANOMALY DETECTION FUNCTION
# ============================================


def find_anomalies(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    anomaly_indices = np.where(np.abs(z_scores) > threshold)[0]
    return anomaly_indices


# Test the function
print("\n--- Testing Function ---")
anomalies = find_anomalies(attempts)
print(f"Anomalies at: {anomalies}")
print(f"Values: {attempts[anomalies]}")
# ============================================

# Tesing pandas and statistics together
# Generate security log
np.random.seed(42)

log = pd.DataFrame(
    {
        "source_ip": np.random.choice(
            ["10.0.0.1", "10.0.0.2", "192.168.1.5", "attacker.evil"],
            200,
            p=[0.3, 0.3, 0.3, 0.1],
        ),
        "bytes": np.random.exponential(1000, 200).astype(int),
        "failed_logins": np.random.poisson(2, 200),
    }
)

# Add one attacker with extreme behavior
log.loc[log["source_ip"] == "attacker.evil", "bytes"] *= 10
log.loc[log["source_ip"] == "attacker.evil", "failed_logins"] += 15

print(log.head(10))
print(f"\nShape: {log.shape}")

# Calculate z-scores for failed_logins
mean_fl = log["failed_logins"].mean()
std_fl = log["failed_logins"].std()

log["fl_zscore"] = (log["failed_logins"] - mean_fl) / std_fl

print(f"Failed logins - Mean: {mean_fl:.2f}, Std: {std_fl:.2f}")
print("\nHighest z-scores:")
print(log.nlargest(5, "fl_zscore")[["source_ip", "failed_logins", "fl_zscore"]])

# Flag rows with z-score > 3
log["is_anomaly"] = np.abs(log["fl_zscore"]) > 3

# Count anomalies per IP
print("Anomalies per IP:")
print(log[log["is_anomaly"]].groupby("source_ip").size())


# Z-score for bytes
mean_bytes = log["bytes"].mean()
std_bytes = log["bytes"].std()
log["bytes_zscore"] = (log["bytes"] - mean_bytes) / std_bytes

log["is_suspicious"] = (np.abs(log["fl_zscore"]) > 3) | (
    np.abs(log["bytes_zscore"]) > 3
)

print("Suspicious events per IP:")
print(log[log["is_suspicious"]].groupby("source_ip").size())


summary = log.groupby("source_ip").agg(
    {"bytes": ["mean", "max"], "failed_logins": ["mean", "max"], "is_suspicious": "sum"}
)

print(summary)


log = pd.DataFrame(
    {"ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3"], "failed_logins": [2, 3, 50]}
)

z_scores = (
    log["failed_logins"] - log["failed_logins"].mean() / log["failed_logins"].std()
)
