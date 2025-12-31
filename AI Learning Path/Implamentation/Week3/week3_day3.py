# Imports
import numpy as np

# Step 1: Create realistic Data
# Y is the threat score
bytes = np.random.randn(100) * 2000 + 5000
duration = np.random.randn(100) * 30 + 60
true_weights = np.array([0.001, 0.05])
X = np.column_stack([bytes, duration])

# Step 2: Scale Data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Step 3: now make Y
Y = np.dot(X_scaled, true_weights)

# Step 4: Verify Shape
print("X shape:", X.shape)
print("Sample threat scores:", Y[:5])

# Step 5: Initialize Model
weights = np.random.randn(2)
learning_rate = 0.01

# Step 6: Build Tradining Loop
for i in range(500):
    predictions = np.dot(X_scaled, weights)
    error = predictions - Y
    loss = np.mean((predictions - Y) ** 2)
    gradient = np.dot(X_scaled.T, error) / len(Y)
    weights = weights - learning_rate * gradient

    if i % 20 == 0:
        print(f"Step {i}: loss = {loss:.4f}")

# Step 7: Display results
print(f"\nTrue weights: {true_weights}")
print(f"Learned Weights: {weights}")
