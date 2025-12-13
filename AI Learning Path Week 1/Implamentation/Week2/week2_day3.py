import numpy as np

# Input: 3 events, 4 features
events = np.array([[0, 100, 0, 0], [10, 50000, 100, 1], [1, 200, 0, 1]])

# Layer 1: 4 features → 3 hidden
W1 = np.array(
    [[0.5, 0.2, 0.1], [0.001, 0.002, 0.001], [0.1, 0.1, 0.3], [0.2, 0.3, 0.1]]
)

# Layer 2: 3 hidden → 2 outputs (attack / not attack)
W2 = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])

# Forward pass
hidden = np.dot(events, W1)
output = np.dot(hidden, W2)

print("Input shape:", events.shape)
print("After layer 1:", hidden.shape)
print("After layer 2:", output.shape)
print("\nOutput (attack probability, safe probability):")
print(output)
