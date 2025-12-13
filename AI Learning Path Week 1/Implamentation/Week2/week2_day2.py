import numpy as np


# 5 events: [failed_logins, bytes_out, port_risk, night_activity]
events = np.array(
    [
        [0, 100, 0, 0],
        [3, 1000, 2, 0],
        [10, 50000, 5, 1],
        [1, 500, 1, 1],
        [8, 30000, 4, 1],
    ]
)

# Weights
weights = np.array([2, 0.001, 1.5, 0.5])

# Your task:
# 1. Calculate scores for all events (one line)
# 2. Print each event's score
# 3. Flag any with score > 10 as "ALERT"

scores = np.dot(events, weights)
print(scores)

for i, score in enumerate(scores):
    if score > 10:
        print(f"Event {i}: {score} - ALERT")
    else:
        print(f"Event {i}: {score} - Ok")
