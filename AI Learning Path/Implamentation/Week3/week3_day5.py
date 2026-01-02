# Imports
import numpy as np

# Create a Classifier that predicts


# Step 1: Create Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Step 2: Test
print(sigmoid(-10))
print(sigmoid(0))
print(sigmoid(10))


# Step 3: Logistics Regression Class
class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, epochs=1000):
        self.weights = np.random.randn(X.shape[1])

        for i in range(epochs):
            z = np.dot(X, self.weights)
            predictions = self.sigmoid(z)
            error = predictions - y
            loss = np.mean((predictions - y) ** 2)
            gradient = np.dot(X.T, error) / len(y)
            self.weights = self.weights - self.learning_rate * gradient

            if i % 100 == 0:
                print(f"Epoch {i}: loss = {loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights)
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)


# Step 4: Create Threat data
# Generate 100 samples
np.random.seed(42)

# Normal traffic (label 0): lower values
normal_bytes = np.random.randn(50) * 1000 + 3000
normal_duration = np.random.randn(50) * 20 + 40

# Threat traffic (label 1): higher values
threat_bytes = np.random.randn(50) * 1000 + 7000
threat_duration = np.random.randn(50) * 20 + 100

# Combine
bytes = np.concatenate([normal_bytes, threat_bytes])
duration = np.concatenate([normal_duration, threat_duration])
X = np.column_stack([bytes, duration])
y = np.array([0] * 50 + [1] * 50)  # 50 normal, 50 threats

print("X shape:", X.shape)
print("y distribution:", np.sum(y == 0), "normal,", np.sum(y == 1), "threats")

# Step 5: Scale and Train
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Step 6: Create Model and Train
model = LogisticRegression(learning_rate=0.01)
model.fit(X_scaled, y, epochs=1000)

# Step 7: Print Results
print("Learned Weights:", model.weights)

# Step 8: Test the model
predictions = model.predict(X_scaled)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100:.1f}%")
print("\nFirst 5 normal (should be 0):", predictions[:5])
print("First 5 threats (should be 1):", predictions[50:55])

# Step 8: Run on new data
new_data = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

probabilities = model.predict_proba(new_data)
predictions = model.predict(new_data)

for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
    label = "Threat" if pred == 1 else "Normal"
    print(f"Sample {i}: probability = {prob:.3f} > {label}")
