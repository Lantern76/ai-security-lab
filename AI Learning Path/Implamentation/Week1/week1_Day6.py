# ////////// Imports and Dictionaries ///////////
import numpy as np

# Classes and Functions

data = np.array([1, 2, 3, 4, 5])


# File data


# Test data


data = np.array([10, 20, 30, 40, 50])

counts = np.array([1, 3, 7, 2, 15, 4, 8])

# Each row: [hour, attempts]
data = np.array([[0, 5], [1, 3], [2, 12], [3, 8], [4, 2]])

attempts = data[:, 1]
peak_index = np.argmax(attempts)
print(peak_index)
print(data[peak_index])  # Full row for that hour


# example

# 24 hours of data: [hour, failed_logins]
hourly_data = np.array(
    [
        [0, 2],
        [1, 1],
        [2, 3],
        [3, 45],
        [4, 52],
        [5, 48],
        [6, 5],
        [7, 3],
        [8, 4],
        [9, 6],
        [10, 5],
        [11, 4],
        [12, 3],
        [13, 5],
        [14, 4],
        [15, 3],
        [16, 5],
        [17, 4],
        [18, 3],
        [19, 2],
        [20, 3],
        [21, 2],
        [22, 1],
        [23, 2],
    ]
)

attempts = hourly_data[:, 1]
peak_hour = np.argmax(attempts)  # Which hour (index)
peak_count = np.max(attempts)  # How many attempts
average = np.mean(attempts)  # Average across all hours


threshold = average * 2
suspicious = attempts > threshold
print(suspicious)


suspicious_hours = hourly_data[suspicious]
print(suspicious_hours)
