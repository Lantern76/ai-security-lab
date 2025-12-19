# Day 4: File I/O & Data Persistence

## Learning Goals
- Read and write files
- Use JSON for structured data
- Handle errors gracefully with try/except
- Build persistent logging system

## Core Concept

**Files are external state.**

Your variables disappear when Python exits. Files persist on disk.

```
Python data → [save] → File on disk → [load] → Python data
```

## Basic File Operations

### Writing to a File

```python
with open("test.txt", "w") as f:
    f.write("Hello, file!")
```

- `open()` — opens a file
- `"w"` — write mode (creates or overwrites)
- `with` — automatically closes file when done
- `as f` — gives you a handle to work with

### Reading from a File

```python
with open("test.txt", "r") as f:
    content = f.read()
    print(content)
```

- `"r"` — read mode

### File Modes

| Mode | Purpose |
|------|---------|
| `"w"` | Write (overwrites existing) |
| `"r"` | Read |
| `"a"` | Append (adds to end) |

## JSON: Structured Data in Files

Python data (lists, dicts) can't be written directly to files. JSON converts them to text.

### Save to JSON

```python
import json

data = {"username": "admin", "ip": "10.0.0.1"}

# To string
text = json.dumps(data)  # '{"username": "admin", "ip": "10.0.0.1"}'

# To file
with open("data.json", "w") as f:
    json.dump(data, f)  # Note: dump, not dumps
```

### Load from JSON

```python
# From string
text = '{"username": "admin", "ip": "10.0.0.1"}'
data = json.loads(text)

# From file
with open("data.json", "r") as f:
    data = json.load(f)  # Note: load, not loads
```

**Memory trick:**
- `dumps` / `loads` = dump **s**tring / load **s**tring
- `dump` / `load` = dump file / load file

## Project: Persistent Security Log

### Save Function

```python
def save_log(log, filename):
    with open(filename, "w") as f:
        json.dump(log, f)
```

### Load Function (Basic)

```python
def load_log(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data
```

## Error Handling

What if the file doesn't exist? Program crashes.

### try/except

```python
def load_log(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return []  # Return empty list if no file
```

### Handle Multiple Errors

```python
def load_log(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return []  # File doesn't exist
    except json.JSONDecodeError:
        print(f"Warning: {filename} is corrupted")
        return []  # File has bad JSON
```

## Project: Append System

Real logs grow over time. Each event gets added:

```python
def append_attempt(filename, username, ip, reason):
    log = load_log(filename)  # Load existing (or empty list)
    log.append({
        "username": username,
        "ip": ip,
        "reason": reason,
        "count": 1
    })
    save_log(log, filename)  # Save updated
```

### Test It

```python
append_attempt("live_log.json", "hacker", "10.0.0.1", "brute force")
append_attempt("live_log.json", "hacker", "10.0.0.1", "brute force")
append_attempt("live_log.json", "admin", "192.168.1.1", "wrong password")

print(load_log("live_log.json"))
# Three events, persisted to disk
```

## Key Concepts Learned

| Concept | Purpose |
|---------|---------|
| `open()` | Open a file |
| `with` | Auto-close file |
| `"w"` / `"r"` / `"a"` | Write / Read / Append modes |
| `json.dump()` | Save Python data to file |
| `json.load()` | Load Python data from file |
| `try/except` | Handle errors gracefully |
| `FileNotFoundError` | File doesn't exist |
| `JSONDecodeError` | File has invalid JSON |

## Success Criteria

- [ ] Can write data to files
- [ ] Can read data from files
- [ ] Can save/load JSON structured data
- [ ] Can handle missing files gracefully
- [ ] Can handle corrupted files gracefully
- [ ] Built persistent append system

## Common Mistakes

1. **`dump` vs `dumps`:** `dump` writes to file, `dumps` returns string
2. **Forgetting `"r"` or `"w"`:** Mode is required in `open()`
3. **Not using `with`:** File might not close properly
4. **Catching wrong exception:** `FileNotFoundError` not `FileNotFound`

## Next Day Preview

Day 5 bundles everything into a class—data and behavior together.
