# Imports

import numpy as np
# Variables


# classes/functions


def normalize(v):
    length = np.linalg.norm(v)
    if length == 0:
        return v
    return v / length


def similarity(v1, v2):
    return np.dot(normalize(v1), normalize(v2))


# Known attack pattern (already scaled 0-5)
# Features: [failed_logins, bytes_out_level, port_risk, night_activity]
attack_signature = np.array(
    [5, 5, 5, 1]
)  # Many failures, high data out, risky ports, at night

# Incoming events to analyze
events = [
    np.array([0, 1, 0, 0]),  # Normal user
    np.array([1, 2, 1, 0]),  # Slightly elevated
    np.array([4, 4, 5, 1]),  # Looks like attack!
    np.array([0, 0, 0, 1]),  # Just night activity
    np.array([5, 5, 4, 1]),  # Very suspicious
]

# Your task: loop through events, calculate similarity to attack_signature
# Print each event's similarity and flag if > 0.9

# Write your code below:

for event in events:
    score = similarity(event, attack_signature)
    if score > 0.9:
        print(f"Score: {score:.3f} - ALERT")
    else:
        print(f"Score: {score:.3f} - OK")


def similarity(v1, v2):
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    results = np.dot(v1_norm, v2_norm)
    return results


events = [
    np.array([1, 2]),
    np.array([3, 4]),
]

# your loop here

for i, event in enumerate(events):
    print(f"Event {i}: {event}")


attack_signature = np.array([5, 5, 5, 1])

events = [
    np.array([1, 2, 3, 4]),
    np.array([1, 1, 1, 1]),
    np.array([0, 0, 0, 0]),
]

for i, event in enumerate(events):
    score = similarity(event, attack_signature)
    if score > 0.9:
        print("ALERT")
    else:
        print("ok")
