# Day 3: Control Flow & Algorithm Thinking

## Learning Goals
- Build multi-level decision trees with `elif`
- Use `set()` for unique values
- Build dictionaries dynamically
- Create a threat detection system

## Core Concept

**Tiered decisions map to security policies.**

```
If 5+ attempts → BLOCK
Else if 3-4 attempts → WARN
Else if 1-2 attempts → MONITOR
Else → OK
```

## Project: Threat Detection System

### Step 1: Threat Level Classifier

```python
def threat_level(count):
    if count >= 5:
        return "Block"
    elif count >= 3:
        return "Warn"
    elif count >= 1:
        return "Monitor"
    else:
        return "OK"
```

`elif` = "else if". Python checks conditions in order, stops at first match.

### Step 2: Assess Single IP

Chain functions together:

```python
def assess_ip(log, ip):
    count = count_by_ip(log, ip)
    level = threat_level(count)
    return level
```

### Step 3: Get Unique Values with `set()`

```python
ips = set()
for attempt in log:
    ips.add(attempt["ip"])
# ips now contains only unique IPs
```

Or one-liner:
```python
unique_ips = {attempt["ip"] for attempt in attempt_log}
```

### Step 4: Scan All IPs

```python
def scan_all_ips(log):
    results = {}  # Empty dictionary
    
    # Get unique IPs
    ips = {attempt["ip"] for attempt in log}
    
    # Assess each
    for ip in ips:
        results[ip] = assess_ip(log, ip)
    
    return results
```

**Building dictionaries:** `results[key] = value` adds a key-value pair.

### Step 5: Filter Blocked IPs

```python
def get_blocked_ips(log):
    threat_map = scan_all_ips(log)
    blocked = []
    for ip, level in threat_map.items():
        if level == "Block":
            blocked.append(ip)
    return blocked
```

**`.items()`** loops through both keys and values of a dictionary.

### Alternative: List Comprehension

```python
def get_blocked_ips(log):
    threat_map = scan_all_ips(log)
    return [ip for ip, level in threat_map.items() if level == "Block"]
```

Same result, one line.

## Simulating an Attack

```python
# Add 5 attempts from same IP
for i in range(5):
    attempt_log.append(log_failed_attempt("hacker", "10.0.0.100", "brute force"))

print(assess_ip(attempt_log, "10.0.0.100"))  # "Block"
```

## Key Concepts Learned

| Concept | Purpose |
|---------|---------|
| `elif` | Multiple conditions in sequence |
| `set()` | Collection of unique values |
| `set.add()` | Add to set |
| `{}` comprehension | Create set in one line |
| `dict[key] = value` | Add to dictionary |
| `.items()` | Loop through dict key-value pairs |
| `range(n)` | Repeat n times |

## The Algorithm Pattern

Every scan/filter function follows:

```python
def scan_something(log):
    results = []  # or {}
    for item in log:
        # analyze item
        # collect result
    return results
```

## Success Criteria

- [ ] Can write `elif` chains for tiered decisions
- [ ] Can extract unique values with `set()`
- [ ] Can build dictionaries dynamically
- [ ] Can loop through dictionary with `.items()`
- [ ] Built working threat detection system

## Common Mistakes

1. **Wrong order in elif:** Check highest threshold first, or lower conditions catch everything
2. **`set()` vs `list()`:** Sets have no duplicates, lists allow duplicates
3. **`dict.items()` forgotten:** Use `.items()` to get both key and value in loop
4. **Building dict wrong:** `results[key] = value`, not `results.append()`

## Next Day Preview

Day 4 adds file persistence—save and load your security logs.
