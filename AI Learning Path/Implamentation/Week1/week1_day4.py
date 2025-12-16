# Imports and Code Dictionarise
import json


# Create test data
attempt_log = []
attempt_log.append(
    {"username": "user", "ip": "172.158.1.50", "reason": "wrong password", "count": 1}
)
attempt_log.append(
    {"username": "admin", "ip": "152.158.1.50", "reason": "wrong password", "count": 1}
)
attempt_log.append(
    {"username": "admin", "ip": "10.0.0.100", "reason": "wrong password", "count": 1}
)


# Funcitons
def save_log(log, filename):
    with open(filename, "w") as f:
        json.dump(log, f)  # Note: dump, not dumps


def load_log(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"Warning: {filename} is corrupted")
        return []


def append_attempt(filename, username, ip, reason):
    log = load_log(filename)
    log.append({"username": username, "ip": ip, "reason": reason, "count": 1})
    save_log(log, filename)


# Examples and singel runs
"""
data = {"username": "admin", "ip": "10.0.0.1"}
text = json.dumps(data)  # dictionary → string
print(text)

text = '{"username": "admin", "ip": "10.0.0.1"}'
data = json.loads(text)  # string → dictionary
print(data["username"])

save_log(attempt_log, "attempts.json")
loaded = load_log("attempts.json")
print(loaded)

with open("bad.json", "w") as f:
    f.write("this is not valid json {{{")

loaded = load_log("bad.json")
loaded = load_log("nonexistent.json")
print(loaded)  # Should print []

loaded = load_log("attempts.json")
print(loaded)  # Should print your data
"""

data = load_log("bad.json")
if data is None:
    print("File corrupted!")
elif data == []:
    print("No data yet")
else:
    print(f"Loaded {len(data)} records")


append_attempt("live_log.json", "hacker", "192.168.1.99", "brute force")
append_attempt("live_log.json", "hacker", "192.168.1.99", "brute force")
append_attempt("live_log.json", "hacker", "192.168.1.99", "brute force")

print(load_log("live_log.json"))
