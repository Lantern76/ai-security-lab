# Day 7: Integration Project

## Learning Goals
- Combine all Week 1 concepts into one tool
- Build a complete security monitoring system
- Practice code organization
- Generate actionable security reports

## Core Concept

**Integration proves understanding.**

You've learned pieces. Now combine them into a working system.

## Project: Security Monitor

A complete tool that:
1. Loads security events from JSON file
2. Analyzes threats using SecurityLog class
3. Detects anomalies using NumPy
4. Outputs actionable threat report

## Step 1: Generate Test Data

Create `generate_data.py`:

```python
import json
import random

events = []
ips = ["192.168.1.10", "10.0.0.50", "172.16.0.5", "10.0.0.100", "192.168.1.99"]
usernames = ["admin", "root", "user", "guest", "administrator"]
reasons = ["wrong password", "account locked", "expired token", "brute force"]

for i in range(100):
    if random.random() < 0.3:
        ip = "10.0.0.100"  # Make this IP appear more often
    else:
        ip = random.choice(ips)
    
    event = {
        "username": random.choice(usernames),
        "ip": ip,
        "reason": random.choice(reasons),
        "hour": random.randint(0, 23)
    }
    events.append(event)

with open("security_events.json", "w") as f:
    json.dump(events, f, indent=2)

print(f"Generated {len(events)} events")
```

## Step 2: Create Security Monitor

Create `security_monitor.py`:

```python
import json
import numpy as np

# ============================================
# CONSTANTS
# ============================================
BLOCK_THRESHOLD = 5
WARN_THRESHOLD = 3

# ============================================
# HELPER FUNCTIONS
# ============================================
def threat_level(count):
    if count >= BLOCK_THRESHOLD:
        return "Block"
    elif count >= WARN_THRESHOLD:
        return "Warn"
    elif count >= 1:
        return "Monitor"
    else:
        return "OK"

# ============================================
# SECURITY LOG CLASS
# ============================================
class SecurityLog:
    def __init__(self):
        self.attempts = []
    
    def add_attempt(self, username, ip, reason):
        attempt = {"username": username, "ip": ip, "reason": reason}
        self.attempts.append(attempt)
    
    def count_by_ip(self, ip):
        matches = []
        for attempt in self.attempts:
            if attempt["ip"] == ip:
                matches.append(attempt)
        return len(matches)
    
    def assess_ip(self, ip):
        count = self.count_by_ip(ip)
        return threat_level(count)
    
    def get_all_ips(self):
        return list({attempt["ip"] for attempt in self.attempts})
    
    def get_blocked_ips(self):
        blocked = []
        for ip in self.get_all_ips():
            if self.assess_ip(ip) == "Block":
                blocked.append(ip)
        return blocked
    
    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.attempts, f)
    
    def load(self, filename):
        try:
            with open(filename, "r") as f:
                self.attempts = json.load(f)
        except FileNotFoundError:
            self.attempts = []
        except json.JSONDecodeError:
            self.attempts = []

# ============================================
# REPORT FUNCTIONS
# ============================================
def generate_report(log):
    print("=== THREAT REPORT ===")
    for ip in log.get_all_ips():
        count = log.count_by_ip(ip)
        level = log.assess_ip(ip)
        print(f"{ip}: {count} attempts - {level}")

def analyze_hourly(log):
    # Extract hours into numpy array
    hours = np.array([attempt["hour"] for attempt in log.attempts])
    
    # Count attempts per hour (0-23)
    hourly_counts = np.zeros(24)
    for h in hours:
        hourly_counts[h] += 1
    
    # Analysis
    peak_hour = np.argmax(hourly_counts)
    peak_count = np.max(hourly_counts)
    average = np.mean(hourly_counts)
    
    # Find anomalies (more than 2x average)
    threshold = average * 2
    anomaly_hours = np.where(hourly_counts > threshold)[0]
    
    print("\n=== HOURLY ANALYSIS ===")
    print(f"Peak hour: {peak_hour}:00 ({int(peak_count)} attempts)")
    print(f"Average per hour: {average:.1f}")
    print(f"Anomaly threshold: {threshold:.1f}")
    print(f"Anomalous hours: {list(anomaly_hours)}")

def run_security_monitor(filename):
    print(f"Loading {filename}...")
    log = SecurityLog()
    log.load(filename)
    print(f"Loaded {len(log.attempts)} events\n")
    
    # Threat report
    generate_report(log)
    
    # Hourly analysis
    analyze_hourly(log)
    
    # Summary
    blocked = log.get_blocked_ips()
    print("\n=== SUMMARY ===")
    print(f"Total events: {len(log.attempts)}")
    print(f"Unique IPs: {len(log.get_all_ips())}")
    print(f"IPs to block: {len(blocked)}")
    print(f"Blocked: {blocked}")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    run_security_monitor("security_events.json")
```

## Step 3: Run and Verify

```bash
python generate_data.py
python security_monitor.py
```

Expected output:
```
Loading security_events.json...
Loaded 100 events

=== THREAT REPORT ===
192.168.1.10: 15 attempts - Block
10.0.0.100: 35 attempts - Block
...

=== HOURLY ANALYSIS ===
Peak hour: 14:00 (8 attempts)
Average per hour: 4.2
Anomaly threshold: 8.3
Anomalous hours: []

=== SUMMARY ===
Total events: 100
Unique IPs: 5
IPs to block: 3
Blocked: ['192.168.1.10', '10.0.0.100', '172.16.0.5']
```

## Code Organization Best Practice

```python
# 1. IMPORTS
import json
import numpy as np

# 2. CONSTANTS
BLOCK_THRESHOLD = 5

# 3. HELPER FUNCTIONS
def threat_level(count):
    ...

# 4. CLASSES
class SecurityLog:
    ...

# 5. MAIN FUNCTIONS
def run_security_monitor(filename):
    ...

# 6. EXECUTION
if __name__ == "__main__":
    run_security_monitor("security_events.json")
```

## What Each Day Contributed

| Day | Component | Used In |
|-----|-----------|---------|
| 1 | Variables, conditionals | Everywhere |
| 2 | Functions, data structures | All functions, SecurityLog |
| 3 | Control flow, threat_level | assess_ip, get_blocked_ips |
| 4 | File I/O, error handling | save(), load() |
| 5 | Classes | SecurityLog |
| 6 | NumPy | analyze_hourly() |

## Success Criteria

- [ ] Can generate test data
- [ ] Can load data into SecurityLog
- [ ] Can generate threat report
- [ ] Can analyze hourly patterns with NumPy
- [ ] Can identify blocked IPs
- [ ] Code is properly organized

## Week 1 Complete

You built a working security tool that:
- Reads from files
- Stores data in objects
- Queries with methods
- Analyzes with NumPy
- Reports actionable intelligence

**This is real engineering, not tutorial-following.**

## Next Week Preview

Week 2: Math + Data Science
- Linear algebra (vectors, matrices)
- NumPy as mathematical notation
- Pandas for data manipulation
- Statistical thinking for ML

The foundation is set. Now we build toward machine learning.
