# ============================================
# 1. IMPORTS (none yet, but they go here)
# ============================================
# example

# ============================================
# 2. CONSTANTS (values that don't change)
# ============================================
BLOCK_THRESHOLD = 5
WARN_THRESHOLD = 3


# ============================================
# 3. FUNCTION DEFINITIONS (all of them, together)
# ============================================
def log_failed_attempt(username, ip, reason):
    attempt = {"username": username, "ip": ip, "reason": reason, "count": 1}
    return attempt


def find_by_username(log, username):
    matches = []  # collect matches here
    for attempt in log:
        if attempt["username"] == username:
            matches.append(attempt)
    return matches


def count_by_ip(log, ip):
    matches = []  # collect matches here
    for attempt in log:
        if attempt["ip"] == ip:
            matches.append(attempt)
    return len(matches)


def threat_level(count):
    if count >= 5:
        return "Block"
    elif 3 <= count <= 4:
        return "Warn"
    elif 1 <= count <= 2:
        return "Monitor"
    else:  # count < 1 (including 0 and negative values)
        return "OK"


def assess_ip(log, ip):
    count = count_by_ip(log, ip)
    level = threat_level(count)
    return level


def scan_all_ips(log):
    results = {}  # empty dictionary to fill

    # Step 1: Get unique IPs
    ips = set()
    for attempt in log:
        ips.add(attempt["ip"])

    # Step 2: Assess each IP
    for ip in ips:
        results[ip] = assess_ip(log, ip)

    return results


def gets_blocked_ips(log):
    threat_map = scan_all_ips(log)
    blocked_ips = [ip for ip, level in threat_map.items() if level == "Block"]
    return blocked_ips


# ============================================
# 4. MAIN EXECUTION (the code that actually runs)
# ============================================
if __name__ == "__main__":
    # Setup
    attempt_log = []

    # Add test data
    attempt_log.append(log_failed_attempt("user", "172.158.1.50", "wrong password"))
    attempt_log.append(log_failed_attempt("admin", "152.158.1.50", "wrong password"))
    attempt_log.append(log_failed_attempt("person", "162.158.1.50", "wrong password"))

    # Simulate attack
    for i in range(5):
        attempt_log.append(log_failed_attempt("admin", "10.0.0.100", "wrong password"))

    # Test

    print(gets_blocked_ips(attempt_log))
