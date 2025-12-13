def log_failed_attempt(username, ip, reason):
    attempt = {"username": username, "ip": ip, "reason": reason, "count": 1}
    return attempt


# Create the list FIRST
attempt_log = []

# Call function and append result (do this 3 times with different data)
attempt_log.append(log_failed_attempt("user", "172.158.1.50", "wrong password"))
# ... two more calls ...
attempt_log.append(log_failed_attempt("admin", "152.158.1.50", "wrong password"))
attempt_log.append(log_failed_attempt("person", "162.158.1.50", "wrong password"))
print(attempt_log)


for attempt in attempt_log:
    if attempt["username"] == "admin":
        print(attempt)


def find_by_username(log, username):
    matches = []  # collect matches here
    for attempt in log:
        if attempt["username"] == username:
            matches.append(attempt)
    return matches


result = find_by_username(attempt_log, "admin")
print(result)

result = find_by_username(attempt_log, "user")
print(result)


def count_by_ip(log, ip):
    matches = []  # collect matches here
    for attempt in log:
        if attempt["ip"] == ip:
            matches.append(attempt)
    return len(matches)


print(count_by_ip(attempt_log, "152.158.1.50"))
print(count_by_ip(attempt_log, "999.999.999.999"))
