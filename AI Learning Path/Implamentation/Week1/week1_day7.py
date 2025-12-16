# Project Data Import
"""
import json
import random

events = []
ips = ["192.168.1.10", "10.0.0.50", "172.16.0.5", "10.0.0.100", "192.168.1.99"]
usernames = ["admin", "root", "user", "guest", "administrator"]
reasons = ["wrong password", "account locked", "expired token", "brute force"]

for i in range(100):
    if random.random() < 0.3:
        ip = "10.0.0.100"
    else:
        ip = random.choice(ips)

    event = {
        "username": random.choice(usernames),
        "ip": ip,
        "reason": random.choice(reasons),
        "hour": random.randint(0, 23),
    }
    events.append(event)

with open("security_events.json", "w") as f:
    json.dump(events, f, indent=2)

print(f"Generated {len(events)} events")
"""

# Imports

import json
import numpy as np

# Constants
BLOCK_THRESHOLD = 5
WARN_THRESHOLD = 3


# Helper function
def threat_level(count):
    if count >= BLOCK_THRESHOLD:
        return "Block"
    elif count >= WARN_THRESHOLD:
        return "Warn"
    elif count >= 1:
        return "Monitor"
    else:
        return "OK"


# SecurityLog class Day 5
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
        level = threat_level(count)
        return level

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.attempts, f)

    def load(self, filename):  # Must be indented here
        try:
            with open(filename, "r") as f:
                self.attempts = json.load(f)
        except FileNotFoundError:
            self.attempts = []
        except json.JSONDecodeError:
            self.attempts = []

    def get_blocked_ips(self):
        blocked = []
        unique_ips = {attempt["ip"] for attempt in self.attempts}
        for ip in unique_ips:
            if self.assess_ip(ip) == "Block":
                blocked.append(ip)
        return blocked

    def get_all_ips(self):
        return list({attempt["ip"] for attempt in self.attempts})


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


# Testing
# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    run_security_monitor("security_events.json")


def find_by_username(self, username):
    # creat an empty list to hold new values
    matches = []
    # need to reference the attempt data
    for attempt in self.attempts:
        # the conditional to find for username
        if attempt["username"] == username:
            # need to append the matches list
            matches.append(attempt)
            # always return data
    return matches


def count_by_reason(self, reason):
    count = []
    for attempt in self.attempts:
        if attempt["reason"] == reason:
            count.append(attempt)
    return len(count)
