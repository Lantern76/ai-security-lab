# ============================================
# IMPORTS
# ============================================
import json

# ============================================
# CONSTANTS
# ============================================
BLOCK_THRESHOLD = 5
WARN_THRESHOLD = 3

# ============================================
# HELPER FUNCTIONS (from previous days)
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
# DAY 5: CLASSES (build here)
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

    # get count using self.count_by_ip()
    # get level using threat_level() (th function)
    # return level

    # same logic, but use self.attempts instead of log parameter


# ============================================
# TESTING (run code here)
# ============================================
if __name__ == "__main__":
    log = SecurityLog()
    print(log.attempts)

# Create and populate
log = SecurityLog()
log.add_attempt("admin", "10.0.0.1", "wrong password")
log.count_by_ip("10.0.0.1")
log.assess_ip("10.0.0.1")
log.save("security.json")
log.load("security.json")

# Create NEW object and load from file


log = SecurityLog()
for i in range(6):
    log.add_attempt("hacker", "10.0.0.100", "brute force")
log.add_attempt("admin", "192.168.1.50", "wrong password")

print(log.get_blocked_ips())
