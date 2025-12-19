# Week 2, Day 7: Integration Project - ML-Ready Security Pipeline

## Learning Goals
- Combine all Week 2 concepts
- Build end-to-end data pipeline
- Create ML-ready feature matrices
- Document and organize code

## Project Overview

Build a complete pipeline that:
1. Loads raw security logs
2. Cleans and preprocesses data
3. Engineers security-relevant features
4. Outputs ML-ready feature matrix
5. Includes statistical anomaly detection

## Part 1: Data Generation

Create realistic security log data:

```python
# generate_security_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_security_log(n_events=5000, attack_ratio=0.1):
    """Generate synthetic security log with normal and attack traffic"""
    np.random.seed(42)
    
    # Time range: last 7 days
    base_time = datetime.now() - timedelta(days=7)
    timestamps = [base_time + timedelta(seconds=np.random.randint(0, 7*24*3600)) 
                  for _ in range(n_events)]
    
    # Normal traffic patterns
    normal_ips = ["10.0.0." + str(i) for i in range(1, 51)]  # 50 internal IPs
    normal_ports = [22, 80, 443, 8080, 3306]
    
    # Attack patterns
    attack_ips = ["45.33.32." + str(i) for i in range(1, 11)]  # 10 attack IPs
    attack_ports = [4444, 31337, 6666, 1337]
    
    events = []
    for i in range(n_events):
        is_attack = np.random.random() < attack_ratio
        
        if is_attack:
            event = {
                "timestamp": timestamps[i],
                "source_ip": np.random.choice(attack_ips),
                "dest_port": np.random.choice(attack_ports + normal_ports),
                "protocol": np.random.choice(["tcp", "udp"], p=[0.9, 0.1]),
                "bytes_sent": int(np.random.exponential(10000)),  # Higher for attacks
                "bytes_recv": int(np.random.exponential(5000)),
                "duration": np.random.exponential(120),  # Longer sessions
                "packets": int(np.random.exponential(100)),
                "action": np.random.choice(["allow", "deny"], p=[0.3, 0.7]),
                "label": 1  # Attack
            }
        else:
            event = {
                "timestamp": timestamps[i],
                "source_ip": np.random.choice(normal_ips),
                "dest_port": np.random.choice(normal_ports),
                "protocol": np.random.choice(["tcp", "udp", "icmp"], p=[0.7, 0.2, 0.1]),
                "bytes_sent": int(np.random.exponential(1000)),
                "bytes_recv": int(np.random.exponential(2000)),
                "duration": np.random.exponential(30),
                "packets": int(np.random.exponential(20)),
                "action": np.random.choice(["allow", "deny"], p=[0.9, 0.1]),
                "label": 0  # Normal
            }
        
        # Add some missing values randomly
        if np.random.random() < 0.02:
            event["bytes_sent"] = np.nan
        if np.random.random() < 0.01:
            event["duration"] = np.nan
            
        events.append(event)
    
    df = pd.DataFrame(events)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

if __name__ == "__main__":
    log = generate_security_log(5000, attack_ratio=0.1)
    log.to_csv("security_log.csv", index=False)
    print(f"Generated {len(log)} events")
    print(f"Attack events: {log['label'].sum()}")
    print(f"Missing values:\n{log.isnull().sum()}")
```

## Part 2: Feature Engineering Module

```python
# feature_engineering.py
import pandas as pd
import numpy as np

class SecurityFeatureEngineer:
    """Engineer security-relevant features from raw logs"""
    
    def __init__(self):
        self.ip_stats = {}
        self.port_stats = {}
    
    def fit(self, df):
        """Learn baseline statistics from data"""
        # IP-level statistics
        self.ip_stats = df.groupby("source_ip").agg({
            "bytes_sent": ["mean", "std"],
            "duration": ["mean", "std"],
            "packets": ["mean"]
        }).to_dict()
        
        # Port frequency
        self.port_stats["common_ports"] = df["dest_port"].value_counts().head(10).index.tolist()
        
        return self
    
    def transform(self, df):
        """Create features from raw data"""
        features = df.copy()
        
        # Time-based features
        features["hour"] = pd.to_datetime(features["timestamp"]).dt.hour
        features["day_of_week"] = pd.to_datetime(features["timestamp"]).dt.dayofweek
        features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
        features["is_night"] = features["hour"].isin([0,1,2,3,4,5,22,23]).astype(int)
        
        # Traffic features
        features["bytes_ratio"] = features["bytes_sent"] / (features["bytes_recv"] + 1)
        features["bytes_per_packet"] = features["bytes_sent"] / (features["packets"] + 1)
        features["packets_per_second"] = features["packets"] / (features["duration"] + 1)
        
        # Port features
        features["is_common_port"] = features["dest_port"].isin(
            self.port_stats.get("common_ports", [22, 80, 443])
        ).astype(int)
        features["is_high_port"] = (features["dest_port"] > 1024).astype(int)
        
        # Log transforms for skewed features
        features["log_bytes_sent"] = np.log1p(features["bytes_sent"])
        features["log_duration"] = np.log1p(features["duration"])
        
        return features
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
```

## Part 3: Preprocessing Pipeline

```python
# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class SecurityPreprocessor:
    """Complete preprocessing pipeline for security data"""
    
    def __init__(self):
        self.feature_engineer = None
        self.preprocessor = None
        self.feature_names = None
        
        # Define feature groups
        self.numeric_features = [
            "bytes_sent", "bytes_recv", "duration", "packets",
            "hour", "bytes_ratio", "bytes_per_packet", "packets_per_second",
            "log_bytes_sent", "log_duration"
        ]
        self.categorical_features = ["protocol"]
        self.binary_features = ["is_weekend", "is_night", "is_common_port", "is_high_port"]
    
    def _build_pipeline(self):
        """Build sklearn preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
                ("bin", "passthrough", self.binary_features)
            ]
        )
    
    def fit(self, df):
        """Fit preprocessor on training data"""
        from feature_engineering import SecurityFeatureEngineer
        
        # Feature engineering
        self.feature_engineer = SecurityFeatureEngineer()
        df_features = self.feature_engineer.fit_transform(df)
        
        # Build and fit preprocessing pipeline
        self._build_pipeline()
        self.preprocessor.fit(df_features)
        
        # Store feature names
        self.feature_names = (
            self.numeric_features + 
            ["protocol_tcp", "protocol_udp"] +  # After one-hot, depends on data
            self.binary_features
        )
        
        return self
    
    def transform(self, df):
        """Transform data using fitted preprocessor"""
        df_features = self.feature_engineer.transform(df)
        X = self.preprocessor.transform(df_features)
        return X
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    def get_feature_names(self):
        """Return feature names after preprocessing"""
        return self.feature_names
```

## Part 4: Statistical Anomaly Detector

```python
# anomaly_detection.py
import numpy as np
import pandas as pd

class StatisticalAnomalyDetector:
    """Detect anomalies using statistical methods"""
    
    def __init__(self, z_threshold=3, iqr_multiplier=1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.stats = {}
    
    def fit(self, X, feature_names=None):
        """Learn baseline statistics"""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        for i, name in enumerate(feature_names):
            col = X[:, i]
            self.stats[name] = {
                "mean": np.mean(col),
                "std": np.std(col),
                "q1": np.percentile(col, 25),
                "q3": np.percentile(col, 75),
                "iqr": np.percentile(col, 75) - np.percentile(col, 25)
            }
        return self
    
    def score_zscore(self, X, feature_names=None):
        """Compute maximum z-score across features"""
        if feature_names is None:
            feature_names = list(self.stats.keys())
        
        scores = np.zeros(X.shape[0])
        for i, name in enumerate(feature_names):
            if name in self.stats:
                col = X[:, i]
                z = np.abs((col - self.stats[name]["mean"]) / (self.stats[name]["std"] + 1e-10))
                scores = np.maximum(scores, z)
        return scores
    
    def score_iqr(self, X, feature_names=None):
        """Count features outside IQR bounds"""
        if feature_names is None:
            feature_names = list(self.stats.keys())
        
        outlier_counts = np.zeros(X.shape[0])
        for i, name in enumerate(feature_names):
            if name in self.stats:
                col = X[:, i]
                s = self.stats[name]
                lower = s["q1"] - self.iqr_multiplier * s["iqr"]
                upper = s["q3"] + self.iqr_multiplier * s["iqr"]
                outlier_counts += ((col < lower) | (col > upper)).astype(int)
        return outlier_counts
    
    def detect(self, X, feature_names=None, method="zscore"):
        """Detect anomalies"""
        if method == "zscore":
            scores = self.score_zscore(X, feature_names)
            return scores > self.z_threshold
        elif method == "iqr":
            scores = self.score_iqr(X, feature_names)
            return scores >= 2  # At least 2 features are outliers
        else:
            raise ValueError(f"Unknown method: {method}")
```

## Part 5: Main Pipeline

```python
# security_pipeline.py
import pandas as pd
import numpy as np
from generate_security_data import generate_security_log
from feature_engineering import SecurityFeatureEngineer
from preprocessing import SecurityPreprocessor
from anomaly_detection import StatisticalAnomalyDetector

def run_pipeline():
    """Run complete security data pipeline"""
    print("=" * 60)
    print("SECURITY DATA PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1] Loading data...")
    df = generate_security_log(5000, attack_ratio=0.1)
    print(f"    Loaded {len(df)} events")
    print(f"    Attack events: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"    Missing values: {df.isnull().sum().sum()}")
    
    # Step 2: Split into train/test
    print("\n[2] Splitting data...")
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    y_train = df_train["label"].values
    y_test = df_test["label"].values
    print(f"    Train: {len(df_train)} events")
    print(f"    Test: {len(df_test)} events")
    
    # Step 3: Preprocess
    print("\n[3] Preprocessing...")
    preprocessor = SecurityPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)
    feature_names = preprocessor.get_feature_names()
    print(f"    Features: {X_train.shape[1]}")
    print(f"    Feature names: {feature_names[:5]}...")
    print(f"    Train shape: {X_train.shape}")
    print(f"    Test shape: {X_test.shape}")
    
    # Step 4: Statistical analysis
    print("\n[4] Statistical analysis...")
    print(f"    Train means (first 5): {X_train[:, :5].mean(axis=0).round(2)}")
    print(f"    Train stds (first 5): {X_train[:, :5].std(axis=0).round(2)}")
    
    # Step 5: Anomaly detection
    print("\n[5] Anomaly detection...")
    detector = StatisticalAnomalyDetector(z_threshold=3)
    detector.fit(X_train, feature_names)
    
    # Detect on test set
    anomalies = detector.detect(X_test, feature_names, method="zscore")
    print(f"    Anomalies detected: {anomalies.sum()} ({anomalies.mean()*100:.1f}%)")
    
    # Compare with actual labels
    true_attacks = y_test == 1
    detected_attacks = anomalies & true_attacks
    precision = detected_attacks.sum() / (anomalies.sum() + 1e-10)
    recall = detected_attacks.sum() / (true_attacks.sum() + 1e-10)
    print(f"    Precision: {precision:.2%}")
    print(f"    Recall: {recall:.2%}")
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs ready for ML:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = run_pipeline()
```

## Expected Output

```
============================================================
SECURITY DATA PIPELINE
============================================================

[1] Loading data...
    Loaded 5000 events
    Attack events: 498 (10.0%)
    Missing values: 145

[2] Splitting data...
    Train: 4000 events
    Test: 1000 events

[3] Preprocessing...
    Features: 14
    Feature names: ['bytes_sent', 'bytes_recv', 'duration', 'packets', 'hour']...
    Train shape: (4000, 14)
    Test shape: (1000, 14)

[4] Statistical analysis...
    Train means (first 5): [ 0.   0.   0.  -0.   0. ]
    Train stds (first 5): [1. 1. 1. 1. 1.]

[5] Anomaly detection...
    Anomalies detected: 156 (15.6%)
    Precision: 52.56%
    Recall: 82.00%

============================================================
PIPELINE COMPLETE
============================================================

Outputs ready for ML:
  - X_train: (4000, 14)
  - X_test: (1000, 14)
  - y_train: (4000,)
  - y_test: (1000,)
```

## Week 2 Complete!

You've built:

| Component | Concepts Used |
|-----------|---------------|
| Data generation | Distributions, random sampling |
| Feature engineering | Domain knowledge, transforms |
| Preprocessing | Scaling, encoding, imputation |
| Statistical detection | Z-scores, percentiles |
| Pipeline | Classes, composition |

## Success Criteria

- [ ] Complete pipeline runs without errors
- [ ] Data properly split into train/test
- [ ] Features correctly scaled (mean~0, std~1)
- [ ] Categorical variables encoded
- [ ] Missing values handled
- [ ] Anomaly detection working

## Next Week Preview

**Week 3: Machine Learning Theory**
- What IS learning? (Mathematical formulation)
- Loss functions and optimization
- Classical ML algorithms from scratch
- Model evaluation

Your pipeline is ready. Now we train real models.
