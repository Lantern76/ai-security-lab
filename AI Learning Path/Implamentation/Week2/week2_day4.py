import numpy as np

# Raw data (list of dictionaries)
attempts = [
    {
        "username": "admin",
        "failed_count": 5,
        "bytes_sent": 1000,
        "hour": 3,
        "is_new_ip": 1,
    },
    {
        "username": "hacker",
        "failed_count": 50,
        "bytes_sent": 99999,
        "hour": 2,
        "is_new_ip": 1,
    },
    {
        "username": "user1",
        "failed_count": 0,
        "bytes_sent": 200,
        "hour": 14,
        "is_new_ip": 0,
    },
]

# Step 1: Extract features into list of lists
features = []
# YOUR LOOP HERE
features = []
for attempt in attempts:
    vector = [
        attempt["failed_count"],
        attempt["bytes_sent"],
        attempt["hour"],
        attempt["is_new_ip"],
    ]
    features.append(vector)
# Step 2: Convert to NumPy array
# YOUR CODE HERE
data = np.array(features)
# Step 3: Define weights and calculate scores
# YOUR CODE HERE
weights = np.array([0.5, 0.0001, 0.3, 0.2])
scores = np.dot(data, weights)
# Step 4: Loop through scores and print ALERT or OK with index
# YOUR CODE HERE
for i, score in enumerate(scores):
    if score > 10:
        print("Alert")
    else:
        print("Ok")
