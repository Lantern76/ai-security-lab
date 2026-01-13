# === WEEK 4 DAY 1: NEURAL NETWORK FROM SCRATCH ===
# Two-layer neural network with backpropagation
# Architecture: Input (2) → Hidden (4) → Output (1)

import numpy as np

# === DATA SETUP ===
np.random.seed(42)

# Normal traffic (label 0)
normal_bytes = np.random.randn(50) * 1000 + 3000
normal_duration = np.random.randn(50) * 20 + 40

# Threat traffic (label 1)
threat_bytes = np.random.randn(50) * 1000 + 7000
threat_duration = np.random.randn(50) * 30 + 100

# Combine into X and y
bytes = np.concatenate([normal_bytes, threat_bytes])
duration = np.concatenate([normal_duration, threat_duration])
X = np.column_stack([bytes, duration])
y = np.array([[0]] * 50 + [[1]] * 50)  # Shape (100, 1)

# Scale the data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# === INITIALIZE WEIGHTS ===
W1 = np.random.randn(2, 4)  # Input to hidden
W2 = np.random.randn(4, 1)  # Hidden to output

learning_rate = 0.1


# === ACTIVATION FUNCTIONS ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(sig_output):
    """Derivative of sigmoid, given the sigmoid output (not z)"""
    return sig_output * (1 - sig_output)


# === TRAINING LOOP ===
for epoch in range(1000):
    # --- Forward Pass ---
    z1 = np.dot(X, W1)          # Input to hidden (100, 4)
    hidden = sigmoid(z1)         # Hidden activations (100, 4)
    z2 = np.dot(hidden, W2)      # Hidden to output (100, 1)
    output = sigmoid(z2)         # Output predictions (100, 1)

    # --- Calculate Error and Loss ---
    output_error = output - y
    loss = np.mean(output_error ** 2)

    # --- Backward Pass ---
    # Output layer gradients
    output_delta = output_error * sigmoid_derivative(output)
    gradient_W2 = np.dot(hidden.T, output_delta) / len(y)

    # Hidden layer gradients (error flows backward through W2)
    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden)
    gradient_W1 = np.dot(X.T, hidden_delta) / len(y)

    # --- Update Weights ---
    W1 = W1 - learning_rate * gradient_W1
    W2 = W2 - learning_rate * gradient_W2

    # --- Print Progress ---
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

# === EVALUATION ===
print("\n" + "=" * 50)
print("=== Final Evaluation ===")

predictions = (output > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100:.1f}%")

print("\nFirst 5 normal (should be 0):", predictions[:5].flatten())
print("First 5 threats (should be 1):", predictions[50:55].flatten())

print("\n=== Network Architecture ===")
print(f"W1 shape: {W1.shape} ({W1.size} parameters)")
print(f"W2 shape: {W2.shape} ({W2.size} parameters)")
print(f"Total parameters: {W1.size + W2.size}")
