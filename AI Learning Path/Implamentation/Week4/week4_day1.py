# Imports
import numpy as np

# Data random
"""
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, (100, 1))
"""
# Step 1: Data Creation
# Data: threat detection (learnable pattern)
np.random.seed(42)

normal_bytes = np.random.randn(50) * 1000 + 3000
normal_duration = np.random.randn(50) * 20 + 40

threat_bytes = np.random.randn(50) * 1000 + 7000
threat_duration = np.random.randn(50) * 30 + 100

bytes = np.concatenate([normal_bytes, threat_bytes])
duration = np.concatenate([normal_duration, threat_duration])
X = np.column_stack([bytes, duration])
y = np.array([[0]] * 50 + [[1]] * 50)

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Initialize weights
W1 = np.random.randn(2, 4)
W2 = np.random.randn(4, 1)

learning_rate = 0.1


def sigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid


def sigmoid_derivative(sig_output):
    return sig_output * (1 - sig_output)


# Step 2: The loop body
for epochs in range(1000):
    # forward pass
    z1 = np.dot(X, W1)
    hidden = sigmoid(z1)
    z2 = np.dot(hidden, W2)
    output = sigmoid(z2)

    # calculate error and loss

    output_error = output - y
    loss = np.mean(output_error**2)

    # Backward pass
    output_delta = output_error * sigmoid_derivative(output)
    gradient_W2 = np.dot(hidden.T, output_delta) / len(y)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden)
    gradient_W1 = np.dot(X.T, hidden_delta) / len(y)

    W1 = W1 - learning_rate * gradient_W1
    W2 = W2 - learning_rate * gradient_W2

    # Print every 100 epochs
    if epochs % 100 == 0:
        print(f"Epochs {epochs}: loss = {loss:.4f}")


# Step 3: Test Predicitons

predictions = (output > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print(f"\nAccuracy: {accuracy * 100:.1f}%")

print("\nFirst 5 normal (should be 0):", predictions[:5].flatten())
print("First 5 threats (should be 1):", predictions[50:55].flatten())
