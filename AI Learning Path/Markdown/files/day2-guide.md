# Day 2: Functions & Data Structures

## Learning Goals
- Wrap logic into reusable functions
- Understand return vs print
- Learn lists vs dictionaries
- Build structured security event logging

## Core Concept

**Functions are reusable transformations.**

```python
def function_name(input):
    # transformation logic
    return output
```

**Critical Rule:** Functions return data. Callers decide what to do with it.

## Project 1: Password Validator Function

### Turn Code Into Function

```python
def validate_password(password):
    failures = []
    
    if len(password) < 8:
        failures.append("at least 8 characters")
    if not any(c.isupper() for c in password):
        failures.append("one uppercase letter")
    if not any(c.islower() for c in password):
        failures.append("one lowercase letter")
    if not any(c.isdigit() for c in password):
        failures.append("one digit")
    if all(c.isalnum() for c in password):
        failures.append("one symbol")
    
    return failures  # Return DATA, not print
```

### Using the Function

```python
result = validate_password("pass")
print(result)  # ['at least 8 characters', 'one uppercase letter', ...]

result = validate_password("SecurePass1!")
print(result)  # [] (empty = all checks passed)
```

## Data Structures

### Lists: Ordered by Position

```python
blocked_ips = ["192.168.1.1", "10.0.0.5"]
blocked_ips[0]  # "192.168.1.1" (first item)
blocked_ips[1]  # "10.0.0.5" (second item)
```

Use lists when:
- Order matters
- Items are similar
- Access by position

### Dictionaries: Accessed by Name

```python
event = {
    "username": "admin",
    "ip": "192.168.1.50",
    "reason": "wrong password"
}
event["ip"]  # "192.168.1.50" (access by key)
```

Use dictionaries when:
- Items have different meanings
- Need named access
- Building structured records

### List of Dictionaries (Common Pattern)

```python
attempt_log = [
    {"username": "admin", "ip": "192.168.1.50", "reason": "wrong password"},
    {"username": "root", "ip": "10.0.0.5", "reason": "account locked"}
]
```

This is how real systems store data.

## Project 2: Security Event Logger

### Create Event Record

```python
def log_failed_attempt(username, ip, reason):
    attempt = {
        "username": username,
        "ip": ip,
        "reason": reason,
        "count": 1
    }
    return attempt
```

### Query the Log

```python
def find_by_username(log, username):
    matches = []
    for attempt in log:
        if attempt["username"] == username:
            matches.append(attempt)
    return matches

def count_by_ip(log, ip):
    matches = []
    for attempt in log:
        if attempt["ip"] == ip:
            matches.append(attempt)
    return len(matches)
```

## Key Pattern: Filter and Collect

Every query function follows this pattern:

```python
def find_by_something(log, value):
    matches = []                          # 1. Empty list
    for item in log:                      # 2. Loop through data
        if item["field"] == value:        # 3. Check condition
            matches.append(item)          # 4. Collect matches
    return matches                        # 5. Return results
```

## Key Concepts Learned

| Concept | Purpose |
|---------|---------|
| `def` | Define a function |
| `return` | Send data back to caller |
| Parameters | Input to function |
| List `[]` | Ordered collection |
| Dictionary `{}` | Key-value pairs |
| `append()` | Add to list |
| `item["key"]` | Access dictionary value |

## Success Criteria

- [ ] Can write functions that return data
- [ ] Understand return vs print
- [ ] Can create and access dictionaries
- [ ] Can query a list of dictionaries
- [ ] Built event logger with query functions

## Common Mistakes

1. **Printing instead of returning:** Functions should `return` data, not `print` it
2. **Keys vs values backwards:** `{"username": username}` not `{username: "username"}`
3. **Variable name collision:** Don't use same name for list and loop variable
4. **Forgetting to create empty list:** Must write `matches = []` before appending

## Next Day Preview

Day 3 adds complex decision trees and threat classification.
