# Week 1: Computational Foundations + Python as Notation

## Starting Point
- Background: Master's in Cybersecurity, CompTIA certified
- Python exposure: Fragmented (pandas, decorators) without solid foundation
- Strength: Theory-first learner, strong systems thinking
- Goal: Build Python as logical expression, not syntax memorization

---

## Day 1: Computation as Transformation

### Core Insight
All computation follows one pattern:
```
Input State → [Transformation Rules] → Output State
```

This maps to security concepts:
- Input validation = controlling what enters
- State management = tracking what system "knows"
- Output handling = controlling what leaves

### Built: Password Validator
Started with security scenario (password policy) and worked backward to code.

**Concepts Learned:**
- Variables hold state
- Conditionals (`if`) make decisions
- Loops via `any()` / `all()` scan sequences
- Lists collect results
- String methods: `isupper()`, `islower()`, `isdigit()`, `isalnum()`

**Key Code Pattern:**
```python
password = "pass123"
failures = []

if len(password) < 8:
    failures.append("at least 8 characters")
if not any(c.isupper() for c in password):
    failures.append("one uppercase letter")
# ... more checks
```

### Design Principle Established
- Design logic BEFORE writing code
- Concept → Logic → Code (never code-first)

---

## Day 2: Functions & Data Structures

### Functions = Reusable Transformations
Wrapped password validator into callable function:
```python
def validate_password(password):
    failures = []
    # ... transformation logic ...
    return failures  # Return DATA, not presentation
```

**Key Principle:** Functions return data. Callers decide presentation.

### Data Structures

**Lists:** Ordered sequences, accessed by position
```python
blocked_ips = ["192.168.1.1", "10.0.0.5"]
blocked_ips[0]  # First item
```

**Dictionaries:** Key-value pairs, accessed by name
```python
attempt = {"username": "admin", "ip": "192.168.1.50", "reason": "wrong password"}
attempt["ip"]  # Access by meaning
```

**List of Dictionaries:** The common pattern for structured data
```python
attempt_log = [
    {"username": "admin", "ip": "192.168.1.50", "reason": "wrong password"},
    {"username": "user", "ip": "172.158.1.50", "reason": "wrong password"}
]
```

### Built: Security Event Logger
Four reusable functions:

| Function | Input | Output | Security Use |
|----------|-------|--------|--------------|
| `validate_password()` | password string | list of failures | Input validation |
| `log_failed_attempt()` | username, ip, reason | dictionary record | Event logging |
| `find_by_username()` | log, username | list of matches | Investigation |
| `count_by_ip()` | log, ip | integer count | Threat detection |

---

## Days 3-7: Where We're Going

### Day 3: Control Flow & Algorithm Thinking
- Nested conditionals
- While loops
- Algorithm design patterns
- Build: Brute-force detection system (threshold alerts)

### Day 4: File I/O & Data Persistence
- Reading/writing files
- CSV and JSON formats
- Error handling with try/except
- Build: Log parser that reads security logs from disk

### Day 5: Object-Oriented Foundations
- Classes as blueprints
- Encapsulating state and behavior
- Build: `SecurityEvent` class, `EventLog` class

### Day 6: NumPy Introduction
- Arrays as mathematical notation
- Vectorized operations
- Why NumPy matters for ML
- Build: Numerical analysis of log patterns

### Day 7: Week 1 Integration Project
- Combine all concepts
- Build: Complete security monitoring tool
  - Reads logs from file
  - Parses into structured data
  - Detects anomalies (brute force, unusual IPs)
  - Outputs alerts

---

## Week 1 Success Criteria

By end of Week 1, you should be able to:

- [ ] Explain computation as input → transformation → output
- [ ] Design logic before writing code
- [ ] Write functions that return data (not print)
- [ ] Choose appropriate data structures (list vs dict)
- [ ] Query and filter structured data
- [ ] Read/write files
- [ ] Build small security tools from scratch

**Meta-skill:** Python feels like notation for ideas you already understand—not arbitrary syntax to memorize.

---

## Code Repository Structure

```
week-01/
├── day1_password_validator.py
├── day2_event_logger.py
├── day3_detection_system.py
├── day4_log_parser.py
├── day5_classes.py
├── day6_numpy_intro.py
└── day7_security_monitor/
    ├── main.py
    ├── parser.py
    ├── detector.py
    └── sample_logs/
```

---

## Notes for Future Reference

**Learning Pattern That Works:**
1. Start with security scenario
2. Design transformation (input → output)
3. Map to Python constructs
4. Build incrementally, testing each piece

**Common Pitfalls Avoided:**
- Variable name mismatches (`failure` vs `failures`)
- Forgetting to call functions (just defining isn't enough)
- Keys vs values in dictionaries
- Returning data vs printing it
