# Imports
import numpy as np

# Todays Project: Take Training Loop from day 3 and wrap it in a class


# Step 1: Define Class
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None

    def fit(self, X, y, epochs=1000):
        self.weights = np.random.randn(X.shape[1])

        for i in range(epochs):
            predictions = np.dot(X, self.weights)
            error = predictions - y
            loss = np.mean((predictions - y) ** 2)
            gradient = np.dot(X.T, error) / len(y)
            self.weights = self.weights - self.learning_rate * gradient

            if i % 20 == 0:
                print(f"Epoch {i}: loss = {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights)


# Step 2: Create data
bytes = np.random.randn(100) * 2000 + 5000
duration = np.random.randn(100) * 30 + 60
X = np.column_stack([bytes, duration])
true_weights = np.array([0.001, 0.05])

# Step 3: Scale Data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Step 4: Create Labels(y)
y = np.dot(X_scaled, true_weights)

# Step 5: Create model and train
model = LinearRegression(learning_rate=0.01)
model.fit(X_scaled, y, epochs=1000)

# Step 6: Check results
print("Learned_weights:", model.weights)
print("True weights:", true_weights)

# Test on new data
new_data = np.array([[0.5, 0.5], [-0.5, -0.5]])  # Already scaled
predictions = model.predict(new_data)
print("Predictions for new data:", predictions)
