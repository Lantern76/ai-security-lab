# Day 1: Computation as Transformation

## Learning Goals
- Understand the core pattern behind all computation
- Learn variables, conditionals, and basic string methods
- Build your first security tool from logic, not syntax

## Core Concept

**All computation follows one pattern:**

```
Input State → [Transformation Rules] → Output State
```

This maps directly to security:
- **Input validation** = controlling what enters the system
- **State management** = tracking what the system "knows"
- **Output handling** = controlling what leaves

## Before You Code

Think about a firewall rule blocking an IP:
- **Input:** Packet with source IP
- **Transformation:** Does IP match blocklist? Yes → drop. No → forward.
- **Output:** Packet dropped or forwarded

This is computation. Everything else is syntax.

## Project: Password Validator

Build a function that checks if a password meets security requirements:
- At least 8 characters
- Contains uppercase letter
- Contains lowercase letter
- Contains digit
- Contains symbol

### Step 1: Store Input

```python
password = "pass123"
```

A variable holds state. That's all it does.

### Step 2: First Check

```python
if len(password) >= 8:
    print("Length OK")
else:
    print("Too short")
```

`len()` measures length. `if/else` makes decisions.

### Step 3: Check Character Types

Python string methods:
- `char.isupper()` → True if uppercase
- `char.islower()` → True if lowercase
- `char.isdigit()` → True if digit
- `char.isalnum()` → True if letter or digit (so `not isalnum()` = symbol)

### Step 4: Scan All Characters

```python
any(char.isupper() for char in password)
```

`any()` returns True if at least one character passes the test.

### Step 5: Collect Failures

```python
failures = []

if len(password) < 8:
    failures.append("at least 8 characters")

if not any(c.isupper() for c in password):
    failures.append("one uppercase letter")

# ... more checks
```

Lists collect results. `append()` adds to the list.

### Step 6: Report Results

```python
if failures:
    print("Password rejected. Missing:", ", ".join(failures))
else:
    print("Password accepted")
```

## Key Concepts Learned

| Concept | Purpose |
|---------|---------|
| Variables | Hold state |
| Conditionals (`if/else`) | Make decisions |
| `len()` | Measure length |
| String methods | Check character types |
| `any()` | Check if any item passes test |
| Lists | Collect multiple items |
| `append()` | Add to list |

## Success Criteria

- [ ] Can explain computation as input → transformation → output
- [ ] Can store data in variables
- [ ] Can write if/else conditionals
- [ ] Can check character types in strings
- [ ] Built working password validator

## Common Mistakes

1. **Unnecessary parentheses:** `password = ("pass123")` → just use `password = "pass123"`
2. **Forgetting `not`:** `any(c.isupper()...)` checks if HAS uppercase. `not any(...)` checks if MISSING.
3. **`isalum()` typo:** It's `isalnum()` (alphanumeric)

## Next Day Preview

Day 2 wraps this logic into reusable functions and introduces data structures.
