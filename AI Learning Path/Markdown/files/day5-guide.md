# Day 5: Object-Oriented Foundations

## Learning Goals
- Understand classes as blueprints
- Bundle data and behavior together
- Use `self` to access object state
- Convert standalone functions to methods

## Core Concept

**Classes bundle data + behavior into one unit.**

Before: Functions and data are separate
```python
attempt_log = []
count_by_ip(attempt_log, ip)  # Pass data every time
```

After: Object contains its own data
```python
log = SecurityLog()
log.count_by_ip(ip)  # Object knows its own data
```

## Analogy: Firewall Appliance

A firewall has:
- **State:** rules, blocked IPs, event log
- **Behavior:** add rule, check packet, log event

You don't pass the firewall's rules into separate functions. The firewall contains and manages its own state.

## Class Syntax

```python
class SecurityLog:
    def __init__(self):
        self.attempts = []  # Object's data
    
    def add_attempt(self, username, ip, reason):
        attempt = {"username": username, "ip": ip, "reason": reason}
        self.attempts.append(attempt)
```

### Key Parts

| Part | Purpose |
|------|---------|
| `class ClassName:` | Define the blueprint |
| `__init__(self)` | Runs when object is created |
| `self` | Reference to this specific object |
| `self.attempts` | This object's data |
| Methods | Functions inside the class |

## Creating and Using Objects

```python
# Create object
log = SecurityLog()

# Use methods
log.add_attempt("admin", "10.0.0.1", "wrong password")
log.add_attempt("admin", "10.0.0.1", "wrong password")

# Access data
print(log.attempts)  # List of 2 attempts
```

## Converting Functions to Methods

### Before (Standalone Function)

```python
def count_by_ip(log, ip):
    matches = []
    for attempt in log:
        if attempt["ip"] == ip:
            matches.append(attempt)
    return len(matches)
```

### After (Method)

```python
def count_by_ip(self, ip):
    matches = []
    for attempt in self.attempts:  # Use self.attempts
        if attempt["ip"] == ip:
            matches.append(attempt)
    return len(matches)
```

**Changes:**
1. Replace `log` parameter with `self`
2. Replace `log` variable with `self.attempts`

## Complete SecurityLog Class

```python
import json

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
        count = self.count_by_ip(ip)  # Call own method with self.
        level = threat_level(count)    # Call outside function (no self.)
        return level
    
    def get_blocked_ips(self):
        blocked = []
        unique_ips = {attempt["ip"] for attempt in self.attempts}
        for ip in unique_ips:
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
```

## Using the Class

```python
# Create and populate
log = SecurityLog()
log.add_attempt("admin", "192.168.1.50", "wrong password")
log.add_attempt("hacker", "10.0.0.100", "brute force")

# Query
print(log.count_by_ip("10.0.0.100"))  # 1
print(log.assess_ip("10.0.0.100"))    # "Monitor"

# Persist
log.save("security.json")

# Load into new object
log2 = SecurityLog()
log2.load("security.json")
print(log2.attempts)  # Same data
```

## When to Use `self.`

| Use `self.` | Don't use `self.` |
|-------------|-------------------|
| Accessing object's data: `self.attempts` | Calling outside functions: `threat_level(count)` |
| Calling object's methods: `self.count_by_ip(ip)` | Local variables: `matches = []` |
| Setting object's data: `self.attempts = []` | Parameters: `ip`, `username` |

## Key Concepts Learned

| Concept | Purpose |
|---------|---------|
| `class` | Define a blueprint |
| `__init__` | Initialize object state |
| `self` | Reference to this object |
| `self.variable` | Object's data |
| Methods | Functions that belong to class |

## Success Criteria

- [ ] Can define a class with `__init__`
- [ ] Can create objects from a class
- [ ] Can write methods that use `self`
- [ ] Can convert standalone functions to methods
- [ ] Built complete SecurityLog class

## Common Mistakes

1. **Forgetting `self.`:** `attempts.append()` → `self.attempts.append()`
2. **Wrong `self.` usage:** `self.threat_level()` when function is outside class
3. **Method outside class:** Indentation wrong—method not inside class block
4. **Forgetting `self` parameter:** Every method needs `self` as first parameter

## Next Day Preview

Day 6 introduces NumPy—operate on entire arrays without loops.
