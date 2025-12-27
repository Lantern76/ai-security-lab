# Weights Flow: Predict, Error, Gradient, Update, Repeat

import numpy as np

# True weights (the model doesn't know these)
true_weights = np.array([3, 2])

# Generate random input data
np.random.seed(42)
X = np.random.randn(100, 2)  # 100 samples, 2 features

# Generate outputs using true weights
y = np.dot(X, true_weights)  # y = X @ true_weights

print("X shape:", X.shape)
print("y shape:", y.shape)
print("First 5 outputs:", y[:5])

# Model's initial guess (random)
weights = np.random.randn(2)
print("Starting weights:", weights)
print("True weights:", true_weights)

# Predict with current weights
predictions = np.dot(X, weights)

# Calculate error (Mean Squared Error)
error = predictions - y
loss = np.mean(error**2)

print("Initial loss:", loss)

# Gradient: direction to reduce loss
gradient = np.dot(X.T, error) / len(y)

print("Gradient:", gradient)

# Learning rate: how big a step to take
learning_rate = 0.1

# Update weights (go opposite of gradient)
weights = weights - learning_rate * gradient

print("Updated weights:", weights)
print("True weights:", true_weights)

# Reset weights
weights = np.random.randn(2)
learning_rate = 0.1

# Learning loop
for epoch in range(50):
    # Forward pass
    predictions = np.dot(X, weights)

    # Calculate loss
    error = predictions - y
    loss = np.mean(error**2)

    # Calculate gradient
    gradient = np.dot(X.T, error) / len(y)

    # Update weights
    weights = weights - learning_rate * gradient

    # Print every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Weights = {weights}")

print("\nFinal weights:", weights)
print("True weights:", true_weights)
