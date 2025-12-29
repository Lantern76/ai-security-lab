"""
Mini Project: Watch Learning Happen
Goal: Build one complete example that shows:

Loss function measuring error
Gradient pointing the direction
Learning rate controlling step size
Weights converging to correct values
"""

import numpy as np

# True weight the model needs to learn
true_weight = 5

# Simple data: 10 threat scores
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# True outputs (X * true_weight)
y = X * true_weight

print("X:", X)
print("y:", y)

# Model's random guess
weight = 0.5
learning_rate = 0.01

# Predict with current weight
predictions = X * weight

# calculate error
error = predictions - y

# calculate loss(MSE)
loss = np.mean((predictions - y) ** 2)

# calculate the gradient
gradient = np.mean(X * error)

# Update the weight
weight = weight - learning_rate * gradient


# Complate training loop
for i in range(100):
    predictions = X * weight
    error = predictions - y
    loss = np.mean((predictions - y) ** 2)
    gradient = np.mean(X * error)
    weight = weight - learning_rate * gradient

    if i % 20 == 0:
        print(f"Step{i}: weight = {weight:.4f}, loss = {loss:.4f}")

print(f"\nFinal weight: {weight:.4f}")
print(f"True weight: {true_weight}")
